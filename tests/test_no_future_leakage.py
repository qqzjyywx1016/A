import pandas as pd

from astock_quant.features.momentum import MomentumFactor
from astock_quant.strategy.selector import StockSelector


def test_momentum_factor_rejects_future_rows_for_signal_date():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "close": 10, "high": 10, "open": 9.8, "low": 9.7, "turnover_amount": 100},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "close": 12, "high": 12, "open": 10, "low": 10, "turnover_amount": 200},
        ]
    )

    try:
        MomentumFactor().calculate(bars, trade_date="2026-06-04")
    except ValueError as exc:
        assert "future" in str(exc).lower()
    else:
        raise AssertionError("factor calculation must reject future rows")


def test_selector_rejects_future_scored_rows():
    scored = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A", "sector_regime": "strong"},
            {"stock_code": "600002.SH", "trade_date": "2026-06-05", "total_score": 95, "rating": "A", "sector_regime": "strong"},
        ]
    )
    selector = StockSelector({"min_total_score": 70, "max_candidates": 20, "max_core_pool": 5})

    try:
        selector.select(scored, trade_date="2026-06-04")
    except ValueError as exc:
        assert "future" in str(exc).lower()
    else:
        raise AssertionError("selector must reject future rows")
