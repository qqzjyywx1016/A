# AStock 策略设计

## 目标

系统每日收盘后运行，使用 T 日及以前可获得的数据，为 T+1 至 T+3 的短线强势延续交易生成候选股、买入建议、仓位建议和风险提示。第一版只做研究、选股、回测和日报，不做自动实盘。

## 数据边界

所有策略计算以 `trade_date <= 信号日` 为硬约束。选股器和因子模块会在计算前按日期截断输入，避免未来函数。T 日收盘后生成信号，最早 T+1 执行买入。回测交易层负责处理停牌、涨停不可买、跌停不可卖、T+1、手续费、印花税和滑点。

数据层通过 `AStockDataAdapter` 统一读取标准 schema。适配器优先读取 `data/processed` 标准化文件；缺失时可调用 `external/a-stock-data/SKILL.md` 中整理出的百度 K 线、东财股票列表、东财板块、东财资金流接口，并在失败时使用 `data/raw/astock_skill` 缓存 fallback。

## 两段式选股框架

第一阶段选股采用两层结构：

1. 硬门槛：股票池过滤、个股 RPS 中期趋势门槛、板块状态门槛、过热过滤和集中度约束。
2. 池内加权排序：只对通过硬门槛的股票用 5 个打分因子计算 `total_score` 并排序。

市场环境不进入 `total_score`，只作为仓位阀门、出场风控和过热过滤条件。当前所有新增权重和阈值均为 provisional, pending IC validation。

## 股票池过滤

过滤条件包括 ST / *ST、停牌、北交所、上市不足 60 个交易日、当日成交额不足 2 亿、近 20 日平均成交额不足 8000 万、近 20 日均换手率不足 1%（列存在时生效）和流通市值小于 30 亿。默认取消流通市值上限；若配置显式给出 `max_float_market_cap`，过滤器仍会执行上限过滤。所有阈值由 `config/default.yaml` 控制，当前为 provisional, pending IC validation。

## 因子体系

第一阶段只使用 5 个打分因子，均输出 0 到 100 分：

- 中期动量：`rps_20`、`rps_60`、趋势效率和 60 日新高。
- 量价：按互斥决策树识别天量长上影、爆量长阴、放量滞涨、放量上涨、缩量回踩和缩量上涨。
- 板块：板块 RPS、板块成交占比、板块内排名和强势股数量。
- 市值弹性：流通市值分层、20 日均成交额分位、20 日均换手率分位和强板块小市值加成。
- 短期反转：有序回踩、下跌减速和缩量质量。

资金结构和事件正向分在第一阶段禁用：`fund_flow` 和 `sentiment` 权重为 0，由 `ScoreEngine` 自动对剩余有效因子重归一。事件数据只作为 Universe 避雷钩子，不作为正向加分。

## RPS 相对价格强度因子

RPS 用横截面排名衡量个股相对全市场的价格强度。系统在完整市场截面上计算 `return_Nd = close / close.shift(N) - 1`，再按每个交易日横截面排名：

```text
rps_N = rank_pct(return_Nd) * 100
```

其中 `rps_5` 反映短线加速，`rps_10` 反映三日策略窗口外的接力持续性，`rps_20` 反映月内主线强度，`rps_60` 反映中期趋势底色。任一交易日有效样本数不足 20 时，该日对应 RPS 使用中性值 50 并记录日志，避免小截面排名误导。

综合 RPS 使用 `0.35*rps_5 + 0.30*rps_10 + 0.25*rps_20 + 0.10*rps_60`。若某只票历史不足导致部分 RPS 缺失，则只用可用分量按剩余权重归一化；全缺时置为 50。相关权重和过滤阈值均从配置读取，当前为 provisional, pending IC validation。

第一阶段动量分改为中期趋势驱动，当前公式为 provisional, pending IC validation：

```text
momentum_score =
  0.35*rps_20
  + 0.25*rps_60
  + 0.25*trend_efficiency_score
  + 0.15*(is_60d_high*100)
```

`trend_efficiency = return_20d / max(volatility_20d, 1e-3)`，其中 `volatility_20d` 是近 20 日日收益率样本标准差。趋势效率先 winsorize，再做横截面 rank score，避免零波动或极端样本导致分数爆炸。

选股阶段会把 RPS 作为硬门槛，但不再使用 `rps_5/rps_10` 入场过滤。新门槛为：`rps_20` 按市场状态分档（strong 65 / neutral 75 / weak 85）、`rps_60 >= 60`、`above_ma20 == True`、`return_10d > -0.03`，且 `sector_regime` 不在 weak/risk_off。`risk_off` 直接不新开仓。这些阈值均从配置读取，当前为 provisional, pending IC validation。`rps_5/rps_10` 只用于过热检查和输出复盘。

RPS 的局限也需要明确：熊市里的高 RPS 可能只是抗跌，并不等于进攻；短期 `rps_5` 很高也可能来自超跌反弹，仍需要成交、板块、资金和形态因子共同确认。

## 板块 RPS 因子

板块 RPS 在 `SectorFactor` 内计算，不改变个股 RPS 逻辑。系统先用板块日线计算 `sector_return_3d/5d/10d/20d = close / close.shift(N) - 1`，再只在信号日对板块横截面排名：

```text
sector_rps_N = rank_pct(sector_return_Nd) * 100
```

`sector_rps_3` 用于识别短线板块加速，`sector_rps_5` 用于确认一周内主线接力，`sector_rps_10` 用于判断持续性，`sector_rps_20` 用于约束中期趋势背景。综合分默认使用 `0.35*sector_rps_3 + 0.30*sector_rps_5 + 0.25*sector_rps_10 + 0.10*sector_rps_20`，缺失分量按可用权重归一化。所有板块 RPS 权重和阈值均来自 `config/default.yaml`，当前为 provisional, pending IC validation。

一股多板块时，系统对每个所属板块合并板块 RPS，并选择 `sector_rps_composite` 最高者作为 `active_sector`。选股阶段可按市场状态应用板块 RPS 硬过滤：强势市场阈值较低，中性市场提高，弱势市场只允许强板块，`risk_off` 不开新仓。板块 RPS 与个股 RPS 中期门槛、趋势门槛和行业集中度过滤取交集，避免只因个股强而忽略板块退潮。

板块 RPS 的局限是板块划分依赖数据源标签质量，概念板块和行业板块可能重叠；高板块 RPS 也可能来自短期消息脉冲，仍需结合成交额、个股排名和情绪状态确认。

## 市值弹性因子

市值弹性因子只在股票池已剔除 30 亿以下流通市值后工作。流通市值分层为 `[30,80)亿 -> 100`、`[80,200)亿 -> 85`、`[200,500)亿 -> 65`、`[500亿,+∞) -> 50`。最终得分为：

```text
market_cap_score =
  0.60*tier_score
  + 0.20*rank_score(avg_turnover_amount_20d)
  + 0.10*rank_score(avg_turnover_rate_20d)
  + 0.10*sector_bonus
```

`sector_bonus` 仅在 active sector 为 strong 且流通市值位于 `[30,80)亿` 时取 100，否则为 0。该分层和权重均为 provisional, pending IC validation。

## 综合评分

默认第一阶段权重为 provisional, pending IC validation：

- 动量 25%
- 量能 20%
- 板块 15%
- 市值弹性 15%
- 短期反转 15%
- 资金 0%
- 情绪 0%

活跃 5 项权重合计 0.90，`ScoreEngine` 会对非零且非中性缺失的因子重归一，保持 `total_score` 量纲稳定。

评级规则：

- A：总分 >= 80
- B：70 <= 总分 < 80
- C：60 <= 总分 < 70
- D：总分 < 60

## 选股输出

默认筛选总分不低于 70、评级为 A/B、板块强度不低于中性，并剔除长上影和放量滞涨标的。观察池最多 20 只，核心池为 A 级且最多 5 只。`selection.max_per_sector=2` 控制最终候选池同一 active sector 数量；回测端同时限制同一 active sector 总仓位不超过 `max_sector_exposure=0.40`。这些阈值均为 provisional, pending IC validation。

## OverheatFilter 前置过热筛选

`OverheatFilter` 是前置筛选器，不是因子，不参与任何加权。完整信号路径为 `UniverseFilter -> 因子计算/打分 -> OverheatFilter -> StockSelector -> TradePlan`。它只使用 T 日收盘后可得字段，在最终选股前剔除短期过热或板块高潮标的，并把被剔除股票保存到 `data/results/YYYY-MM-DD_rejected.csv`，字段包含 `reject_reason`。

过热规则包括个股 RPS_5 高位大涨放量、RPS_5/10 高潮加速、爆量滞涨和板块高潮。所有阈值均来自 `config.default.yaml` 的 `overheat` 段，当前为 provisional, pending IC validation。RPS_5/RPS_10 在这里仅用于入场前过热检查；它们高位维持或短期回落都不代表持仓见顶，不能作为卖出条件。

## 买卖规则

买入建议只生成交易计划，不自动下单。次日竞价涨幅 0% 到 5% 为竞价确认，高开大于 7% 或一字涨停回避，盘中突破昨日高点且放量为突破买入，回踩 5/10 日线且缩量不破为低吸。

卖出规则由配置驱动。优先级为止损、移动止损、止盈、趋势破位、板块退潮和最大持仓天数兜底。默认配置将固定 3 天持仓调整为 10 天上限、止盈提高到 20%、移动止损 8%、趋势破位使用 MA10；这些参数为 provisional, pending backtest validation。代码在缺少这些配置时保持旧行为，便于测试和兼容。

## ContinueHoldScore 持仓退出

持仓退出使用 `ContinueHoldScore` 评估普通退出，不使用 RPS_5/RPS_10 作为卖出条件。评分范围 0 到 10，由趋势、RPS_20 中期动量、板块状态、量价健康和风险状态五项组成。`RPS_20/RPS_60` 更适合描述中期趋势背景，`RPS_5/RPS_10` 只用于入场筛选和过热检查。

退出优先级为 hard_stop_loss、market_risk_off、major_event_risk、trailing_stop、take_profit、ContinueHoldScore、max_holding_days。只有当 ContinueHoldScore 所需指标齐全时才评估普通退出，分数低于配置阈值才触发 `low_continue_hold_score`；指标缺失时回退到旧的规则引擎路径。相关阈值来自配置，当前为 provisional, pending IC/backtest validation。

## 回测假设

回测以日线为主，T 日信号在 T+1 开盘价附近成交。买入价加入正向滑点，卖出价扣除滑点。买入收手续费，卖出收手续费和印花税。涨停日不可买入，跌停日不可卖出，停牌日不可交易。

同一 active sector 回测持仓数量默认不超过 2 只，且同一 active sector 总市值占当前权益默认不超过 40%。单票仓位默认 15%，总仓位由市场状态阀门控制：strong 80%、neutral 50%、weak 20%、risk_off 0%。这些参数均为 provisional, pending IC/backtest validation。

## IC 验证工具

`scripts/analyze_factor_ic.py` 是研究脚本，与信号路径隔离。脚本按日计算 5 个因子分和 `total_score`，再用未来 1/3/5/10 日收益做事后 Spearman Rank IC 评估，输出 `data/results/factor_ic_report.csv` 和 `reports/factor_ic_report.md`。forward return 只用于事后 IC 验证，绝不进入选股、打分或回测信号生成路径。

## 日报

日报输出市场状态、情绪周期、强势板块、Top 20 候选股、核心股说明、风险提示和明日交易计划，用于盘后复盘和次日交易准备。
