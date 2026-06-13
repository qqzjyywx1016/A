Baidu 日线接口状态: DEAD(VPN on/off 均 403);当前数据源 = baostock(前复权 adjustflag=2)

# 数据可得性评估

| 数据项 | 当前状态 | 来源/口径 | 缺口 | 处理建议 |
| --- | --- | --- | --- | --- |
| 历史退市股票池 | 包含 | baostock `query_stock_basic`, `status=0` | 退市覆盖减轻幸存者偏差, 但仍需复核历史 ST/PIT 状态 | 保留退市股票进入历史数据抓取 |
| 逐日 qfq | 有 | baostock `query_history_k_data_plus`, `adjustflag=2` | 依赖 baostock 前复权口径 | 写入 `adjust_type=qfq`, 作为回测准入硬条件 |
| 逐日流通市值 | 无 | baostock 免费源未提供 point-in-time `circ_mv` | `market_cap` 因子与 30 亿流通市值门槛降级 | 后续接 tushare `circ_mv` 修复 |
| 稳定行业 PIT 成分 | 部分 | baostock 申万行业, 当前成分, 无 `effective_date` | 存在板块后见之明风险 | 回测默认只允许 `industry` 类型, 概念板块关闭 |
| 中证1000/国证2000 | 中证1000 有, 国证2000 视接口可得性 | baostock 指数日线 | 国证2000 不可得时规模基准退化为中证1000 | 报告中标注基准退化 |

若仅当前在册股票:幸存者偏差,结果为乐观上界(尤其高估小盘动量)

市值口径降级为已知局限: 当前 baostock 阶段不伪造逐日流通市值, `universe.min_float_market_cap` 与 `score_weights.market_cap` 已置空/置零, 暂以 20 日平均成交额与 20 日平均换手率作为可交易性代理。后续接入 tushare `circ_mv` 后恢复市值门槛和市值弹性因子。
