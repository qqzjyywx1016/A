# AStock

AStock 是一个每日收盘后运行的 A 股强势股三日动量选股系统。第一版聚焦选股、规则打分、回测和 Markdown 日报，不包含自动实盘交易。

## 项目目标

- 从全市场过滤出可交易股票池，剔除 ST、停牌、北交所、流动性不足和事件风险标的。
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

## 在 Windows 上从零运行(含 baostock 连通性测试)

本节针对“在 Mac 上写代码、复制到 Windows 上跑数据/回测”的场景。Windows 不开 VPN，更适合连 baostock(中国服务器)。

### 0. Python 版本(重要)

baostock 0.8.9 与 **Python 3.13 可能不兼容**(import 或连接报错)。Windows 上请安装 **Python 3.11 或 3.12**:
- 从 https://www.python.org/downloads/ 下载 3.11.x 或 3.12.x；
- 安装时务必勾选 **“Add python.exe to PATH”**;
- 安装完在 PowerShell 里确认:`py -0p`(列出已装的 Python)。

### 1. 从 Mac 打包(只拷源码,不拷环境和数据)

`.venv` 是平台相关的二进制,**绝对不能拷到 Windows**;`data/` 里的 parquet 也别拷(口径可能过期),到 Windows 重新生成。Mac 上执行:

```bash
cd ..
zip -r astock_to_windows.zip astock_momentum_selector \
  -x "*/.venv/*" -x "*/__pycache__/*" -x "*/.pytest_cache/*" \
  -x "*.egg-info/*" -x "*.pyc" \
  -x "*/data/processed/*" -x "*/data/raw/*" -x "*/data/results/*"
```

把 `astock_to_windows.zip` 传到 Windows,解压到例如 `C:\quant\astock_momentum_selector`。

### 2. Windows 上创建环境(PowerShell)

```powershell
cd C:\quant\astock_momentum_selector

# 用 3.11 建虚拟环境(没有 3.11 就把版本号换成你装的 3.12)
py -3.11 -m venv .venv

# 激活(PowerShell)。若提示脚本被禁止,先执行下一行的 Set-ExecutionPolicy,再重试激活
.\.venv\Scripts\Activate.ps1
# 如被拦:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   然后回答 Y,再跑上面的激活
# 或者改用 CMD 激活:  .\.venv\Scripts\activate.bat

python -m pip install -U pip
pip install -e ".[dev]"
```

### 3. 验证安装

```powershell
pytest -q
```
应看到 `86 passed`(数量随版本变化)。能跑通说明依赖装好了。

### 4. baostock 连通性测试(这是搬到 Windows 的核心目的)

```powershell
python -c "import baostock as bs; r=bs.login(); print('login:', r.error_code, r.error_msg); bs.logout()"
```
- 打印 `login: 0 success`(error_code 为 `0`)→ **连通**,继续第 5 步。
- 报错 / 长时间挂起 / error_code 非 0 → 这台 Windows 也连不上 baostock,告诉我,我们再换路子(可能要换 tushare 付费积分,或换数据源)。

### 5. 小样本数据冒烟(连通后)

先只拉几只蓝筹、半年,别一上来拉全市场:

```powershell
python scripts\ingest_baostock.py --start 2024-01-01 --end 2024-06-30 --codes sh.600519,sz.000001,sh.600000
```
检查生成的数据是否合理:
```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/processed/daily_bars.parquet'); print(df.shape); print(df[['stock_code','trade_date','close','pct_chg','adjust_type']].head(8))"
```
重点看:`adjust_type` 是否为 `qfq`;`pct_chg` 是否为小数(如 0.03 而非 3.0);`close` 是否是前复权价(连续、无除权跳变)。

### 6. 拉全市场(放着过夜跑,带限速)

冒烟没问题后,**不传 `--codes` 就是全市场**:脚本用 `query_all_stock` 枚举全市场(含历年退市股,减轻幸存者偏差),逐只拉前复权日线。全市场约 5000+ 只,每只要查 basic / daily / industry 三次,适合晚上挂着跑。

为避免被限制,加了限速参数(全部可选,默认不休眠以兼容小样本):

- `--sleep`:每只票请求之间停顿秒数(如 `0.5`)。
- `--batch-size` + `--batch-rest`:每拉 N 只就长休一次(如每 200 只休 60 秒),即"拉一会休息一会"。
- `--jitter`:把每次停顿随机抖动 ±该比例(默认 0.2),让节奏不固定。

建议的过夜命令(慢而稳,从 2022 年起覆盖完整熊市与两次风格切换):

```powershell
python scripts\ingest_baostock.py --start 2022-01-01 --end 2026-06-30 --sleep 0.6 --batch-size 200 --batch-rest 60 --relogin-every 500
```

**关于掉线自动重连(重要):** baostock 不是 HTTP 接口,而是一条 `bs.login()` 建立的 TCP 长连接。长时间全市场拉取时这条连接会被服务器/网络掐断,表现为 `WinError 10054 远程主机强迫关闭连接` 或 `10002007 网络接收错误`,而且会**连续多只票一起失败**——这是会话失活,不是逐请求限流,也和 User-Agent / `requests` 无关(baostock 走私有 socket 协议,没有 HTTP 头可改)。脚本已内置应对:

- **掉线自动重连**:查询遇到连接类错误时,会先 `logout` + `login` 重建会话再重试,而不是在死连接上空重试。
- **主动定期重连**:`--relogin-every N` 每拉 N 只票主动刷新一次会话,在它失活前就换新连接(预防为主)。
- 真的某些票反复失败,会被记入末尾失败汇总并跳过,不影响整体。

**断点续传已内置**:中途断了、被限了、或电脑睡眠了,直接**重跑同一条命令**即可——已拉全的票会自动跳过(`skip ... existing coverage`),`--save-every`(默认 50)会定期落盘,不会白拉。跑完看末尾打印的 `daily_bars rows=...` 和失败汇总。

### 7. 数据质检准入门槛

```powershell
python scripts\validate_real_data.py --start 2022-01-01 --end 2026-06-30
```
打印 `OK: data validation passed hard gates` 才允许进入后续回测;`BLOCKED` 则按提示修数据。(小样本验证仍用对应的小区间日期。)

### 常见问题

- **激活报“无法加载文件 Activate.ps1,因为在此系统上禁止运行脚本”**:执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 后重试,或改用 `activate.bat`。
- **baostock 装不上或 import 报错**:多半是 Python 3.13,改装 3.11/3.12 重建 `.venv`。
- **baostock 全市场 `query_stock_basic()` 在部分网络会挂起**:本脚本已改为逐只查询 + `query_all_stock` 枚举,故首次全市场 ingest 仍会较慢(逐只),但不会卡死;冒烟用 `--codes` 指定几只会秒级完成。
- **路径**:脚本内部用 `pathlib`,跨平台没问题;只是命令行里 Windows 用反斜杠 `scripts\xxx.py`。
- **不要把 Mac 的 `.venv`、`data/*.parquet` 拷过来**:venv 重建,数据重新 ingest。

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

### 数据源: baostock 阶段的降级说明

新增 `scripts/ingest_baostock.py` 可从 baostock 拉取前复权日线并写入 `data/processed` 标准 parquet。日线使用 `adjustflag=2`，并写入 `adjust_type=qfq`；指数使用 `adjustflag=3`。除 `is_limit_up/is_limit_down` 外，落库时还会从 low/high 相对昨收的比例派生 `is_sealed_limit_up/is_sealed_limit_down` 封板标志（比例在复权下不变），回测引擎优先消费这些标志而不是用 qfq 价格推算绝对涨停价。可用小样本先冒烟：

```bash
python scripts/ingest_baostock.py --start 2026-06-01 --end 2026-06-04 --codes sh.600000,sz.000001
```

baostock 免费源不提供逐日流通市值。当前配置显式将 `universe.min_float_market_cap` / `max_float_market_cap` 置空，并将 `score_weights.market_cap` 置为 0.0；ScoreEngine 会对其余有效因子权重归一化。这是 provisional, pending IC validation 的阶段性降级，暂用 `min_avg_turnover_amount_20d` 与 `min_avg_turnover_rate_20d` 作为可交易性代理，后续接 tushare `circ_mv` 后再恢复市值门槛和市值弹性因子。

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

回测脚本读取 `data/results/signals.csv`，因此需要先运行批量信号脚本。批量信号脚本还会生成 `data/results/backtest_panel.parquet`，供 ContinueHoldScore 使用；面板不存在或某日字段缺失（NaN）时回测自动回退到规则退出引擎，不会基于缺失数据强制卖出。信号中的 `suggested_position`（按评级生成）会被回测直接用作开仓比例。回测每日先卖后买（卖出资金当日可用），支持最低佣金、高开回避、组合回撤熔断，指标包含夏普/索提诺/年化波动/回撤修复天数与跳空止损统计。交易记录和每日净值会保存到 `data/results`。

## 如何生成日报

```bash
python scripts/generate_report.py --date 2026-06-04
```

输出文件为 `reports/YYYY-MM-DD_daily_report.md`。

## 配置

默认配置位于 `config/default.yaml`。路径、股票池阈值、打分权重、RPS 权重与阈值、过热过滤阈值、选股数量、回测参数和仓位参数均从配置读取，策略逻辑中不写死路径。当前新增的 RPS、板块 RPS、过热过滤、ContinueHoldScore、趋势退出和移动止损参数为 provisional，pending IC/backtest validation。
