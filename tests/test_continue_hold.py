import pandas as pd

from astock_quant.backtest.engine import BacktestEngine
from astock_quant.strategy.continue_hold import ContinueHoldScorer


def _healthy_row(**overrides) -> pd.Series:
    row = {
        "close": 12.0,
        "ma5": 11.5,
        "ma10": 11.0,
        "rps_5": 82,
        "rps_10": 80,
        "rps_20": 85,
        "sector_rps_5": 75,
        "sector_rps_10": 70,
        "high_volume_bearish": False,
        "high_volume_stagnation": False,
        "long_upper_shadow": False,
        "market_regime": "strong",
        "is_major_event": False,
    }
    row.update(overrides)
    return pd.Series(row)


def test_rps5_pullback_does_not_trigger_exit_when_continue_hold_score_strong():
    decision = ContinueHoldScorer({}).evaluate(_healthy_row(rps_5=82, rps_10=78))

    assert decision.score >= 8
    assert decision.action == "strong_hold"


def test_close_below_ma5_above_ma10_only_deducts_trend_score():
    decision = ContinueHoldScorer({}).evaluate(_healthy_row(close=11.2, ma5=11.5, ma10=11.0))

    assert decision.components["trend_score"] == 1
    assert decision.action in {"strong_hold", "hold_watch"}


def test_low_continue_hold_score_exits():
    row = _healthy_row(
        close=10,
        ma5=11,
        ma10=10.5,
        rps_20=50,
        sector_rps_5=40,
        sector_rps_10=35,
        high_volume_stagnation=True,
        market_regime="weak",
    )

    decision = ContinueHoldScorer({}).evaluate(row)

    assert decision.score < 6
    assert decision.action == "exit"
    assert decision.reason == "low_continue_hold_score"


def test_backtest_continue_hold_exits_but_does_not_use_rps5_drop():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.5, "low": 9.8, "close": 10.4, "turnover_amount": 1_000_000, "ma5": 10, "ma10": 9.8, "rps_5": 98, "rps_10": 96, "rps_20": 86, "sector_rps_5": 75, "sector_rps_10": 70, "high_volume_stagnation": False, "long_upper_shadow": False, "market_regime": "strong", "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 10.4, "high": 10.8, "low": 10.2, "close": 10.6, "turnover_amount": 1_000_000, "ma5": 10.3, "ma10": 10.1, "rps_5": 82, "rps_10": 80, "rps_20": 85, "sector_rps_5": 75, "sector_rps_10": 70, "high_volume_stagnation": False, "long_upper_shadow": False, "market_regime": "strong", "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-09", "open": 10.6, "high": 10.7, "low": 10.4, "close": 10.5, "turnover_amount": 1_000_000, "ma5": 10.7, "ma10": 10.8, "rps_5": 70, "rps_10": 68, "rps_20": 50, "sector_rps_5": 40, "sector_rps_10": 35, "high_volume_stagnation": True, "long_upper_shadow": False, "market_regime": "weak", "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": None,
            "stop_loss": -0.20,
            "take_profit": 0.50,
            "continue_hold": {"enabled": True, "exit_below": 6},
        }
    ).run(bars, signals).trades

    sells = trades[trades["side"] == "SELL"]
    assert sells.iloc[0]["trade_date"] == "2026-06-09"
    assert sells.iloc[0]["reason"] == "low_continue_hold_score"


def test_backtest_exit_priority_before_continue_hold():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.5, "low": 9.8, "close": 10.2, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 10.2, "high": 10.3, "low": 10.0, "close": 10.1, "turnover_amount": 1_000_000, "ma5": 10.2, "ma10": 10.1, "rps_20": 90, "sector_rps_5": 80, "sector_rps_10": 75, "market_regime": "risk_off", "is_major_event": True, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": None,
            "stop_loss": -0.20,
            "take_profit": 0.50,
            "continue_hold": {"enabled": True, "exit_below": 6},
        }
    ).run(bars, signals).trades

    assert trades[trades["side"] == "SELL"].iloc[0]["reason"] == "market_risk_off"
