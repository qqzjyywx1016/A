# a-stock-data 真实接口适配说明

## 适配边界

`external/a-stock-data/SKILL.md` 是带内嵌 Python 代码的 Skill 文档，不是可直接 import 的 Python package。本项目没有把策略层绑定到该文档，而是新增 `astock_quant/data/astock_skill_sources.py`，将其中可用端点整理为 `AStockSkillSource`。

`AStockDataAdapter` 的读取顺序：

1. 优先读取 `data/processed/*.parquet|csv` 标准化文件。
2. 本地标准文件缺失且 `external.live_enabled=true` 时调用 `AStockSkillSource`。
3. 外部接口失败或返回空时读取 `data/raw/astock_skill/*.parquet` 缓存。
4. 仍无数据时返回空表并记录日志。

## 已实现端点

| 标准方法 | 外部来源 | 标准化字段 |
| --- | --- | --- |
| `get_stock_basic()` | 东财 `push2 clist/get` | `stock_code`, `stock_name`, `exchange`, `is_st`, `sector`, `total_market_cap`, `float_market_cap` |
| `get_daily_bars()` | 百度股市通 `selfselect/getstockquotation` | `open`, `high`, `low`, `close`, `volume`, `amount`, `turnover_amount`, `pct_chg` |
| `get_index_bars()` | 百度股市通指数 K 线 fallback | 沪深300、中证1000、上证指数、深证成指、创业板指 |
| `get_sector_map()` | 东财 `push2 slist/get` | 股票主板块、板块代码、板块涨跌幅、全部概念标签 |
| `get_sector_daily()` | 东财行业板块 `push2 clist/get` | 板块涨跌幅、成交额、上涨/下跌家数、领涨股 |
| `get_fund_flow()` | 东财 `push2his stock/fflow/daykline/get` | 主力、超大单、大单净流入 |
| `get_trading_calendar()` | 指数或行情日期推导 | `trade_date`, `is_open` |
| `get_limit_status()` | 日线 fallback 计算 | `limit_up`, `limit_down`, `is_limit_up`, `is_limit_down` |

## 标准化约定

- 股票代码统一为 `000001.SZ` / `600000.SH` / `832000.BJ`。
- `trade_date` 使用 pandas normalized timestamp。
- `daily_bars` 和 `index_bars` 的价格字段必须按前复权 `qfq` 口径提供。适配器读到本地标准文件后只做疑似未复权异常跳变日志提示，不在策略层自动改价。
- 成交额字段同时保留 `amount` 和 `turnover_amount`，策略层使用 `turnover_amount`。
- 涨跌幅 `pct_chg` 使用百分数，例如 `4.2` 表示上涨 4.2%。
- 板块日涨跌幅 `sector_return_1d` 使用小数，例如 `0.025` 表示上涨 2.5%。

## TODO

- 百度指数 K 线在部分指数代码上可能返回空；需要根据实盘验证补充指数专用接口。
- `stock_basic.list_date` 需要逐股调用东财个股信息接口才能补齐，当前默认不批量补齐，避免首次全市场更新过慢。
- `sector_daily` 当前是东财行业板块当日快照，不是历史板块 K 线；`sector_return_3d` 暂用当日涨跌幅 fallback。
- `fund_flow.main_net_inflow_ratio` 目前无成交额分母时为 0，可在后续与日线成交额合并后计算。
