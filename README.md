# AStock

AStock 是一个每日收盘后运行的 A 股强势股三日动量选股系统。第一版聚焦选股、规则打分、回测和 Markdown 日报，不包含自动实盘交易。

## 项目目标

- 从全市场过滤出可交易股票池，剔除 ST、停牌、北交所、流动性不足和市值不匹配标的。
- 计算动量、量能、板块、资金、形态、情绪六类规则因子，并支持个股 RPS 与板块 RPS。
- 使用非机器学习的权重打分模型输出核心池和观察池。
- 回测 T 日收盘信号在 T+1 执行的短线策略，并考虑涨跌停、停牌、T+1、手续费、印花税和滑点。
- 生成每日选股 Markdown 报告。

## 安装方式

```bash
cd astock_momentum_selector
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 数据源说明

外部数据源按要求放置在：

```text
external/a-stock-data
```

目标仓库为 `https://github.com/simonlin1212/a-stock-data`。策略代码不直接依赖外部项目实现细节，而是通过 `MarketDataAdapter` 和 `AStockDataAdapter` 访问数据。

当前数据层优先读取本项目 `data/processed` 下的标准 parquet/csv 文件；如果本地标准文件不存在且 `external.live_enabled=true`，会调用 `AStockSkillSource` 中从 `external/a-stock-data/SKILL.md` 提取的真实接口。默认配置中 `external.live_enabled=false`，需要实时接口时请显式打开，避免新环境首次运行触发外部接口风控。

- 百度股市通 K 线：标准化为 `daily_bars` / `index_bars`
- 东财 clist：标准化为 `stock_basic` 和行业板块快照
- 东财 slist：标准化为 `sector_map`
- 东财 push2his 资金流：标准化为 `fund_flow`
- 交易日历：从指数或行情日期推导
- 涨跌停状态：优先使用行情字段，缺失时按 ST 5%、创业板/科创板 20%、主板 10% 粗略计算

外部接口失败时会优先读取 `data/raw/astock_skill/*.parquet` 缓存；缓存也不存在时返回空表并写日志，不会让策略层崩溃。

标准行情契约：`daily_bars` 和 `index_bars` 的价格字段必须使用前复权 `qfq` 口径。适配器读取本地标准文件时会对疑似未复权的异常跳变写 warning，但不会自动修改价格。

## 如何更新数据

```bash
python scripts/update_data.py
```

`update_data.py` 会通过 `AStockDataAdapter.update_data()` 拉取真实接口数据，并保存为本项目标准 parquet 文件。默认会写入：

- `data/processed/stock_basic.parquet`
- `data/processed/daily_bars.parquet`
- `data/processed/index_bars.parquet`
- `data/processed/sector_map.parquet`
- `data/processed/sector_daily.parquet`
- `data/processed/fund_flow.parquet`
- `data/processed/trading_calendar.parquet`

批量拉全市场会触发大量 HTTP 请求，尤其是东财接口有风控。可在 `config/default.yaml` 的 `external.astock_codes` 或 `external.max_fetch_codes` 限制首次拉取范围。

## 如何运行选股

```bash
python scripts/run_selection.py --date 2026-06-04
```

输出保存到 `data/results/YYYY-MM-DD_selection.csv`。
若启用 `overheat`，过热前置筛选剔除的股票会保存到 `data/results/YYYY-MM-DD_rejected.csv`，用于复盘。

## 如何运行回测

```bash
python scripts/run_batch_signals.py --start 2023-01-01 --end 2026-06-04
python scripts/run_backtest.py --start 2023-01-01 --end 2026-06-04
```

回测脚本读取 `data/results/signals.csv`，因此需要先运行批量信号脚本。批量信号脚本还会生成 `data/results/backtest_panel.parquet`，供 ContinueHoldScore 使用；面板不存在时回测会自动回退。交易记录和每日净值会保存到 `data/results`。

## 如何生成日报

```bash
python scripts/generate_report.py --date 2026-06-04
```

输出文件为 `reports/YYYY-MM-DD_daily_report.md`。

## 配置

默认配置位于 `config/default.yaml`。路径、股票池阈值、打分权重、RPS 权重与阈值、过热过滤阈值、选股数量、回测参数和仓位参数均从配置读取，策略逻辑中不写死路径。当前新增的 RPS、板块 RPS、过热过滤、ContinueHoldScore、趋势退出和移动止损参数为 provisional，pending IC/backtest validation。
