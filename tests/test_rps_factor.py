import logging

import pandas as pd
import pytest

from astock_quant.features.momentum import MomentumFactor
from astock_quant.strategy.selector import StockSelector
from astock_quant.utils.errors import DataQualityError


def _rps_bars(stock_count: int = 21, periods: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2026-02-27", periods=periods)
    rows = []
    for stock_index in range(stock_count):
        code = f"{stock_index + 1:06d}.SZ"
        strength = stock_index + 1
        for day_index, trade_date in enumerate(dates):
            close = 100 + day_index * strength
            rows.append(
                {
                    "stock_code": code,
                    "trade_date": trade_date,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "turnover_amount": 100_000_000 + stock_index,
                }
            )
    return pd.DataFrame(rows)


def _selector_row(**overrides):
    row = {
        "stock_code": "000001.SZ",
        "stock_name": "强势股份",
        "trade_date": "2026-06-04",
        "sector": "机器人",
        "total_score": 88,
        "rating": "A",
        "sector_regime": "strong",
        "market_regime": "strong",
        "long_upper_shadow": False,
        "high_volume_stagnation": False,
        "rps_5": 76,
        "rps_10": 71,
        "rps_20": 76,
        "rps_60": 65,
        "rps_composite": 72,
        "return_5d": 0.05,
        "return_10d": -0.02,
        "above_ma20": True,
    }
    row.update(overrides)
    return row


def _selector_config():
    return {
        "min_total_score": 70,
        "max_candidates": 20,
        "max_core_pool": 5,
        "rps": {
            "enabled": True,
            "filters": {
                "strong": {"rps_20": 65},
                "neutral": {"rps_20": 75},
                "weak": {"rps_20": 85},
                "risk_off": {"allow_new_position": False},
            },
            "gate": {
                "rps_60_min": 60,
                "return_10d_min": -0.03,
            },
        },
    }


def test_rps_factor_outputs_required_columns_and_highest_return_has_highest_rps():
    bars = _rps_bars()
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    result = MomentumFactor().calculate(bars, trade_date=trade_date)

    required = {
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
        "rps_5",
        "rps_10",
        "rps_20",
        "rps_60",
        "rps_composite",
        "ma20",
        "above_ma20",
        "rps_pattern",
        "is_20d_high",
        "is_60d_high",
    }
    assert required.issubset(result.columns)
    strongest = result.sort_values("return_5d").iloc[-1]
    assert strongest["rps_5"] == 100
    assert strongest["rps_composite"] == 100


def test_rps_factor_rejects_future_rows():
    bars = pd.DataFrame(
        [
            {"stock_code": "000001.SZ", "trade_date": "2026-06-04", "close": 10, "high": 10},
            {"stock_code": "000001.SZ", "trade_date": "2026-06-05", "close": 11, "high": 11},
        ]
    )

    with pytest.raises(ValueError, match="future"):
        MomentumFactor().calculate(bars, trade_date="2026-06-04")


def test_rps_factor_uses_neutral_rps_when_cross_section_too_small(caplog):
    bars = _rps_bars(stock_count=3, periods=70)
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    with caplog.at_level(logging.WARNING):
        result = MomentumFactor().calculate(bars, trade_date=trade_date)

    assert set(result["rps_5"]) == {50.0}
    assert "valid_count=3" in caplog.text
    assert caplog.text.count("RPS fallback to neutral") == 4


def test_rps_composite_uses_configured_weights_and_available_components():
    bars = _rps_bars()
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    result = MomentumFactor().calculate(bars, trade_date=trade_date)
    row = result[result["stock_code"] == "000011.SZ"].iloc[0]
    expected = (
        0.35 * row["rps_5"]
        + 0.30 * row["rps_10"]
        + 0.25 * row["rps_20"]
        + 0.10 * row["rps_60"]
    )

    assert row["rps_composite"] == round(expected, 2)


def test_selector_applies_regime_specific_rps_thresholds():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame(
        [
            _selector_row(market_regime="strong"),
            _selector_row(stock_code="000002.SZ", market_regime="weak"),
        ]
    )

    selected = selector.select(rows, trade_date="2026-06-04")

    assert selected["watch_pool"]["stock_code"].tolist() == ["000001.SZ"]


def test_selector_does_not_gate_on_rps5_rps10_or_positive_return10():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame([_selector_row(rps_5=10, rps_10=10, rps_20=90, rps_60=80, return_10d=-0.02)])

    selected = selector.select(rows, trade_date="2026-06-04")

    assert selected["watch_pool"]["stock_code"].tolist() == ["000001.SZ"]


def test_selector_returns_empty_in_risk_off_regime():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame([_selector_row(market_regime="risk_off", rps_20=100, rps_60=100)])

    selected = selector.select(rows, trade_date="2026-06-04")

    assert selected["watch_pool"].empty
    assert selected["core_pool"].empty


def test_selector_raises_data_quality_error_when_rps_columns_missing():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame([_selector_row()]).drop(columns=["rps_20"])

    with pytest.raises(DataQualityError, match="rps_20"):
        selector.select(rows, trade_date="2026-06-04")


def test_selector_requires_rps60_gate_and_return10_floor():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame(
        [
            _selector_row(stock_code="000001.SZ", rps_60=59, return_10d=0.05),
            _selector_row(stock_code="000002.SZ", rps_60=70, return_10d=-0.031),
            _selector_row(stock_code="000003.SZ", rps_60=70, return_10d=-0.02),
        ]
    )

    selected = selector.select(rows, trade_date="2026-06-04")

    assert selected["watch_pool"]["stock_code"].tolist() == ["000003.SZ"]


def test_selector_requires_explicit_true_above_ma20():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame([_selector_row(above_ma20=float("nan"))])

    selected = selector.select(rows, trade_date="2026-06-04")

    assert selected["watch_pool"].empty


def test_momentum_sector_relative_return_ignores_sector_code_only_data():
    bars = _rps_bars(stock_count=21, periods=70)
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")
    sector_map = pd.DataFrame(
        [{"stock_code": code, "sector_code": "BK001"} for code in bars["stock_code"].drop_duplicates()]
    )
    sector_daily = pd.DataFrame(
        [
            {"sector_code": "BK001", "trade_date": trade_date, "close": 100},
        ]
    )

    result = MomentumFactor().calculate(
        bars,
        trade_date=trade_date,
        sector_map=sector_map,
        sector_daily=sector_daily,
    )

    assert "relative_return_vs_sector" in result.columns
    assert set(result["relative_return_vs_sector"]) == {0.0}
