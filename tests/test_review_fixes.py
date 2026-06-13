"""Regression tests for the quant-review fixes.

Covers: ContinueHold NaN handling, sparse-panel backtests, same-day sell-then-buy
cash reuse, gap-up avoidance, the portfolio drawdown circuit breaker, minimum
commission, rating-based sizing, new risk metrics, gap-stop marking, sentiment
regime smoothing/absolute panic gate/index trend filter, sealed limit flags,
buy-plan generation, and score-cliff smoothing.
"""

import math

import pandas as pd
import pytest

from astock_quant.backtest.engine import BacktestEngine
from astock_quant.features.reversal import ReversalFactor
from astock_quant.features.sentiment import SentimentFactor
from astock_quant.features.volume import long_upper_shadow_flag
from astock_quant.strategy.buy_rules import BuyRuleEngine
from astock_quant.strategy.continue_hold import ContinueHoldScorer
from scripts.ingest_baostock import derive_limit_flags


def _bar(code, date, **kwargs):
    row = {
        "stock_code": code,
        "trade_date": date,
        "open": 10.0,
        "high": 10.1,
        "low": 9.9,
        "close": 10.0,
        "turnover_amount": 100_000_000,
        "is_limit_up": False,
        "is_limit_down": False,
        "is_suspended": False,
    }
    row.update(kwargs)
    return row


def _engine_config(**overrides):
    config = {
        "initial_cash": 100_000,
        "commission_rate": 0,
        "stamp_tax_rate": 0,
        "slippage_rate": 0,
        "max_holding_days": None,
        "stop_loss": -0.99,
        "take_profit": 9.99,
        "max_participation": 1.0,
        "single_position_pct": 0.5,
    }
    config.update(overrides)
    return config


def test_continue_hold_can_evaluate_rejects_nan_rows():
    row = pd.Series(
        {
            "close": 10.0,
            "ma5": float("nan"),
            "ma10": float("nan"),
            "rps_20": float("nan"),
            "sector_rps_5": float("nan"),
            "sector_rps_10": float("nan"),
            "market_regime": float("nan"),
        }
    )

    assert ContinueHoldScorer({}).can_evaluate(row) is False


def test_continue_hold_nan_flags_do_not_count_as_risk():
    row = pd.Series(
        {
            "close": 12.0,
            "ma5": 11.5,
            "ma10": 11.0,
            "rps_20": 85,
            "sector_rps_5": 75,
            "sector_rps_10": 70,
            "market_regime": "strong",
            "high_volume_bearish": float("nan"),
            "high_volume_stagnation": float("nan"),
            "long_upper_shadow": float("nan"),
            "is_major_event": float("nan"),
        }
    )

    decision = ContinueHoldScorer({}).evaluate(row)

    assert decision.components["volume_price_health_score"] == 2
    assert decision.components["risk_score"] == 2
    assert decision.action == "strong_hold"


def test_backtest_sparse_panel_does_not_force_sell_healthy_position():
    """A held stock missing from the scored panel must fall back to rule exits."""

    dates = ["2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10"]
    closes = [10.0, 10.2, 10.4, 10.6, 10.8]
    rows = []
    for index, (date, close) in enumerate(zip(dates, closes, strict=True)):
        extra = {}
        if index == 1:
            # Panel covers only the signal date; later days have NaN factor fields.
            extra = {"ma5": 10.1, "rps_20": 86, "sector_rps_5": 75, "sector_rps_10": 70, "market_regime": "strong"}
        rows.append(_bar("600001.SH", date, open=close, high=close + 0.1, low=close - 0.1, close=close, **extra))
    bars = pd.DataFrame(rows)
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        _engine_config(continue_hold={"enabled": True, "exit_below": 6})
    ).run(bars, signals).trades

    assert "low_continue_hold_score" not in trades["reason"].tolist()
    assert trades[trades["side"] == "SELL"].empty


def test_backtest_sell_proceeds_available_for_same_day_buy():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04"),
            _bar("600001.SH", "2026-06-05"),
            _bar("600001.SH", "2026-06-08", open=12, high=12.2, low=11.8, close=12),
            _bar("600002.SH", "2026-06-04"),
            _bar("600002.SH", "2026-06-05"),
            _bar("600002.SH", "2026-06-08"),
        ]
    )
    signals = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"},
            {"stock_code": "600002.SH", "trade_date": "2026-06-05", "total_score": 90, "rating": "A"},
        ]
    )

    trades = BacktestEngine(
        _engine_config(max_positions=1, take_profit=0.10, single_position_pct=0.95)
    ).run(bars, signals).trades

    day3 = trades[trades["trade_date"] == "2026-06-08"]
    assert set(day3["side"]) == {"SELL", "BUY"}
    assert day3[day3["side"] == "BUY"].iloc[0]["stock_code"] == "600002.SH"


def test_backtest_avoids_excessive_gap_up_open():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04"),
            _bar("600001.SH", "2026-06-05", open=10.8, high=10.9, low=10.6, close=10.7),
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    blocked = BacktestEngine(
        _engine_config(buy_rules={"avoid_gap_up_pct": 0.07})
    ).run(bars, signals).trades
    allowed = BacktestEngine(_engine_config()).run(bars, signals).trades

    assert blocked.empty
    assert not allowed[allowed["side"] == "BUY"].empty


def test_backtest_drawdown_circuit_breaker_blocks_new_entries():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04"),
            _bar("600001.SH", "2026-06-05"),
            _bar("600001.SH", "2026-06-08", open=7.2, high=7.3, low=6.9, close=7.0),
            _bar("600002.SH", "2026-06-04"),
            _bar("600002.SH", "2026-06-05"),
            _bar("600002.SH", "2026-06-08"),
        ]
    )
    signals = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"},
            {"stock_code": "600002.SH", "trade_date": "2026-06-05", "total_score": 90, "rating": "A"},
        ]
    )

    blocked = BacktestEngine(
        _engine_config(max_portfolio_drawdown=0.10)
    ).run(bars, signals).trades
    allowed = BacktestEngine(
        _engine_config(max_portfolio_drawdown=0.50)
    ).run(bars, signals).trades

    assert "600002.SH" not in blocked[blocked["side"] == "BUY"]["stock_code"].tolist()
    assert "600002.SH" in allowed[allowed["side"] == "BUY"]["stock_code"].tolist()


def test_backtest_minimum_commission_applies_to_small_orders():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04", turnover_amount=30_000),
            _bar("600001.SH", "2026-06-05", turnover_amount=30_000),
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        _engine_config(commission_rate=0.0003, min_commission=5, max_participation=0.05)
    ).run(bars, signals).trades

    buy = trades[trades["side"] == "BUY"].iloc[0]
    assert buy["amount"] < 5 / 0.0003
    assert buy["fee"] == 5.0


def test_backtest_uses_suggested_position_from_signal():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04"),
            _bar("600001.SH", "2026-06-05"),
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "trade_date": "2026-06-04",
                "total_score": 90,
                "rating": "A",
                "market_regime": "strong",
                "suggested_position": 0.10,
            }
        ]
    )

    trades = BacktestEngine(_engine_config(single_position_pct=0.5)).run(bars, signals).trades

    buy = trades[trades["side"] == "BUY"].iloc[0]
    assert buy["amount"] == pytest.approx(10_000, abs=1_000)


def test_metrics_include_risk_ratios_and_nan_excess_without_benchmark():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04"),
            _bar("600001.SH", "2026-06-05"),
            _bar("600001.SH", "2026-06-08", open=10.4, high=10.6, low=10.3, close=10.5),
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])
    engine = BacktestEngine(_engine_config())

    metrics = engine.run(bars, signals).metrics
    benchmark = pd.DataFrame({"trade_date": ["2026-06-04", "2026-06-08"], "equity": [100_000.0, 101_000.0]})
    metrics_with_benchmark = engine.run(bars, signals, benchmark_curve=benchmark).metrics

    for key in ["sharpe_ratio", "sortino_ratio", "annual_volatility", "drawdown_recovery_days", "gap_stop_count", "gap_stop_share"]:
        assert key in metrics
    assert math.isnan(metrics["benchmark_excess_return"])
    assert not math.isnan(metrics_with_benchmark["benchmark_excess_return"])


def test_backtest_marks_gap_stop_loss_exits():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04"),
            _bar("600001.SH", "2026-06-05"),
            _bar("600001.SH", "2026-06-08", open=9.0, high=9.1, low=8.8, close=9.0),
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    result = BacktestEngine(_engine_config(stop_loss=-0.05)).run(bars, signals)

    sell = result.trades[result.trades["side"] == "SELL"].iloc[0]
    assert sell["reason"] == "stop_loss"
    assert sell["price"] == 9.0
    assert sell["gap_exit"] is True
    assert result.metrics["gap_stop_count"] == 1


def _sentiment_bars(day_ratios: dict[str, float], stock_count: int = 10) -> pd.DataFrame:
    rows = []
    for date, ratio in day_ratios.items():
        up_count = round(ratio * stock_count)
        for index in range(stock_count):
            up = index < up_count
            rows.append(
                {
                    "stock_code": f"{index + 1:06d}.SZ",
                    "trade_date": date,
                    "close": 11.0 if up else 9.0,
                    "prev_close": 10.0,
                    "turnover_amount": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_sentiment_confirm_days_smooths_single_day_crash():
    bars = _sentiment_bars({"2026-06-02": 1.0, "2026-06-03": 1.0, "2026-06-04": 0.3})

    smoothed = SentimentFactor({"confirm_days": 3}).calculate(bars, trade_date="2026-06-04")
    raw = SentimentFactor({"confirm_days": 1}).calculate(bars, trade_date="2026-06-04")

    assert smoothed.iloc[0]["market_regime"] == "strong"
    assert raw.iloc[0]["market_regime"] == "weak"


def test_sentiment_risk_off_requires_absolute_limit_down_count():
    bars = _sentiment_bars({"2026-06-04": 0.2})
    limit_status = pd.DataFrame(
        {"stock_code": ["a", "b", "c", "d"], "is_limit_up": [True, False, False, False], "is_limit_down": [False, True, True, True]}
    )

    calm = SentimentFactor({"risk_off_min_limit_down": 30}).calculate(
        bars, trade_date="2026-06-04", limit_status=limit_status
    )
    panic = SentimentFactor({"risk_off_min_limit_down": 2}).calculate(
        bars, trade_date="2026-06-04", limit_status=limit_status
    )

    assert calm.iloc[0]["market_regime"] == "weak"
    assert panic.iloc[0]["market_regime"] == "risk_off"


def test_sentiment_strong_requires_index_above_ma20():
    bars = _sentiment_bars({"2026-06-04": 0.7})
    dates = pd.bdate_range("2026-05-01", "2026-06-04").strftime("%Y-%m-%d")
    rising = pd.DataFrame({"index_code": "000300.SH", "trade_date": dates, "close": range(100, 100 + len(dates))})
    falling = pd.DataFrame({"index_code": "000300.SH", "trade_date": dates, "close": range(100 + len(dates), 100, -1)})

    strong = SentimentFactor({}).calculate(bars, trade_date="2026-06-04", index_bars=rising)
    capped = SentimentFactor({}).calculate(bars, trade_date="2026-06-04", index_bars=falling)

    assert strong.iloc[0]["market_regime"] == "strong"
    assert capped.iloc[0]["market_regime"] == "neutral"


def test_ingest_derives_sealed_limit_flags_from_ratios():
    frame = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "pct_chg": 0.10, "prev_close": 10.0, "low": 11.0, "high": 11.0},
            {"stock_code": "600002.SH", "pct_chg": 0.10, "prev_close": 10.0, "low": 10.5, "high": 11.0},
            {"stock_code": "600003.SH", "pct_chg": -0.10, "prev_close": 10.0, "low": 9.0, "high": 9.0},
        ]
    )

    result = derive_limit_flags(frame)

    assert result["is_sealed_limit_up"].tolist() == [True, False, False]
    assert result["is_sealed_limit_down"].tolist() == [False, False, True]


def test_backtest_prefers_explicit_sealed_flags():
    bars = pd.DataFrame(
        [
            _bar("600001.SH", "2026-06-04"),
            _bar(
                "600001.SH",
                "2026-06-05",
                open=11,
                high=11,
                low=11,
                close=11,
                is_limit_up=True,
                is_sealed_limit_up=True,
            ),
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(_engine_config()).run(bars, signals).trades

    assert trades.empty


def test_buy_rule_engine_generates_executable_price_levels():
    candidates = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "close": 10.0,
                "high": 10.4,
                "ma5": 9.9,
                "ma10": 9.7,
                "rps_pattern": "acceleration",
            },
            {
                "stock_code": "600002.SH",
                "close": 20.0,
                "high": 20.4,
                "ma5": 19.8,
                "ma10": 19.5,
                "rps_pattern": "neutral",
            },
        ]
    )

    result = BuyRuleEngine({"auction_confirm_min_pct": 0.0, "auction_confirm_max_pct": 0.05, "avoid_gap_up_pct": 0.07}).generate(candidates)

    first = result.iloc[0]
    assert first["suggestion"] == "breakout_buy_plan"
    assert first["buy_zone_low"] == 10.0
    assert first["buy_zone_high"] == 10.5
    assert first["avoid_above"] == 10.7
    assert first["breakout_level"] == 10.4
    assert result.iloc[1]["suggestion"] == "auction_confirm_plan"


def test_buy_rule_engine_handles_empty_candidates():
    result = BuyRuleEngine({}).generate(pd.DataFrame())

    assert "suggestion" in result.columns
    assert result.empty


def test_reversal_shrink_score_is_smooth_at_old_cliff():
    dates = pd.bdate_range("2026-05-20", periods=11)
    rows = []
    for code, last_amount in [("600001.SH", 69.0), ("600002.SH", 71.0)]:
        for index, date in enumerate(dates):
            amount = last_amount if index == len(dates) - 1 else 100.0
            close = 10.0 + index * 0.05
            rows.append(
                {
                    "stock_code": code,
                    "trade_date": date,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "turnover_amount": amount,
                }
            )

    result = ReversalFactor().calculate(pd.DataFrame(rows), trade_date=dates[-1].strftime("%Y-%m-%d")).set_index("stock_code")

    assert abs(result.loc["600001.SH", "shrink_score"] - result.loc["600002.SH", "shrink_score"]) < 10


def test_long_upper_shadow_not_triggered_by_bearish_body():
    open_price = pd.Series([10.4, 10.0])
    high = pd.Series([10.45, 10.6])
    low = pd.Series([9.9, 9.95])
    close = pd.Series([9.95, 10.05])

    flags = long_upper_shadow_flag(open_price, high, low, close)

    # Big bearish body is not an upper shadow; a true shadow above a small body is.
    assert flags.tolist() == [False, True]
