import logging

import pandas as pd
import pytest

from astock_quant.features.sector import SectorFactor
from astock_quant.strategy.selector import StockSelector
from astock_quant.utils.errors import DataQualityError


def _sector_config(min_cross_section: int = 5) -> dict:
    return {
        "enabled": True,
        "min_cross_section": min_cross_section,
        "composite_weights": {
            "sector_rps_3": 0.35,
            "sector_rps_5": 0.30,
            "sector_rps_10": 0.25,
            "sector_rps_20": 0.10,
        },
        "filters": {
            "strong": {"sector_rps_5": 55, "sector_rps_10": 50},
            "neutral": {"sector_rps_5": 60, "sector_rps_10": 55},
            "weak": {"sector_rps_5": 75, "sector_rps_10": 70},
            "risk_off": {"allow_new_position": False},
        },
    }


def _sector_fixture(sector_count: int = 6, periods: int = 25) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-05-01", periods=periods)
    sector_rows = []
    map_rows = []
    bar_rows = []
    for sector_index in range(sector_count):
        sector_code = f"BK{sector_index + 1:03d}"
        strength = sector_index + 1
        for day_index, trade_date in enumerate(dates):
            close = 100 + day_index * strength
            sector_rows.append(
                {
                    "sector_code": sector_code,
                    "sector_name": f"板块{sector_index + 1}",
                    "sector_type": "industry",
                    "trade_date": trade_date,
                    "close": close,
                    "turnover_amount": 100_000_000 * strength,
                }
            )
        code = f"600{sector_index + 1:03d}.SH"
        map_rows.append(
            {
                "stock_code": code,
                "sector_code": sector_code,
                "sector_name": f"板块{sector_index + 1}",
                "sector_type": "industry",
            }
        )
        for day_index, trade_date in enumerate(dates):
            close = 10 + day_index * (sector_index + 1) * 0.1
            bar_rows.append(
                {
                    "stock_code": code,
                    "trade_date": trade_date,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "turnover_amount": 50_000_000 * strength,
                }
            )
    return pd.DataFrame(bar_rows), pd.DataFrame(map_rows), pd.DataFrame(sector_rows)


def _selector_row(**overrides):
    row = {
        "stock_code": "600001.SH",
        "stock_name": "强势股份",
        "trade_date": "2026-06-04",
        "total_score": 88,
        "rating": "A",
        "sector": "机器人",
        "sector_regime": "strong",
        "market_regime": "strong",
        "long_upper_shadow": False,
        "high_volume_stagnation": False,
        "rps_5": 90,
        "rps_10": 88,
        "rps_20": 85,
        "rps_60": 75,
        "rps_composite": 88,
        "return_5d": 0.08,
        "return_10d": 0.12,
        "above_ma20": True,
        "sector_rps_5": 56,
        "sector_rps_10": 51,
        "sector_rps_composite": 58,
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
            },
            "gate": {"rps_60_min": 60, "return_10d_min": -0.03},
        },
        "sector_rps": _sector_config(),
    }


def test_sector_rps_outputs_fields_and_highest_sector_has_top_rank():
    bars, sector_map, sector_daily = _sector_fixture()
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    result = SectorFactor(_sector_config()).calculate(
        bars,
        trade_date=trade_date,
        sector_map=sector_map,
        sector_daily=sector_daily,
    )

    required = {
        "sector_return_3d",
        "sector_return_5d",
        "sector_return_10d",
        "sector_return_20d",
        "sector_rps_3",
        "sector_rps_5",
        "sector_rps_10",
        "sector_rps_20",
        "sector_rps_composite",
        "sector_rps_pattern",
        "active_sector_code",
        "active_sector_name",
        "active_sector_type",
        "active_sector_rps",
    }
    assert required.issubset(result.columns)
    strongest = result.sort_values("sector_return_5d").iloc[-1]
    assert strongest["sector_rps_5"] == 100
    assert strongest["sector_rps_composite"] == 100


def test_sector_rps_uses_only_signal_date_cross_section_warning(caplog):
    bars, sector_map, sector_daily = _sector_fixture(sector_count=3)
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    with caplog.at_level(logging.WARNING):
        result = SectorFactor(_sector_config(min_cross_section=5)).calculate(
            bars,
            trade_date=trade_date,
            sector_map=sector_map,
            sector_daily=sector_daily,
        )

    assert set(result["sector_rps_5"]) == {50.0}
    assert "valid_count=3" in caplog.text
    assert caplog.text.count("Sector RPS fallback to neutral") == 4


def test_sector_rps_composite_and_acceleration_pattern():
    bars, sector_map, sector_daily = _sector_fixture()
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    result = SectorFactor(_sector_config()).calculate(
        bars,
        trade_date=trade_date,
        sector_map=sector_map,
        sector_daily=sector_daily,
    )
    row = result[result["active_sector_code"] == "BK006"].iloc[0]
    expected = (
        0.35 * row["sector_rps_3"]
        + 0.30 * row["sector_rps_5"]
        + 0.25 * row["sector_rps_10"]
        + 0.10 * row["sector_rps_20"]
    )

    assert row["sector_rps_composite"] == round(expected, 2)
    assert row["sector_rps_pattern"] == "sector_acceleration"


def test_multi_sector_stock_uses_highest_composite_as_active_sector():
    bars, sector_map, sector_daily = _sector_fixture()
    extra_map = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "sector_code": "BK006",
                "sector_name": "板块6",
                "sector_type": "concept",
            }
        ]
    )
    sector_map = pd.concat([sector_map, extra_map], ignore_index=True)
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    result = SectorFactor(_sector_config()).calculate(
        bars,
        trade_date=trade_date,
        sector_map=sector_map,
        sector_daily=sector_daily,
    )

    row = result[result["stock_code"] == "600001.SH"].iloc[0]
    assert row["active_sector_code"] == "BK006"
    assert row["active_sector_name"] == "板块6"
    assert row["active_sector_type"] == "concept"
    assert row["active_sector_rps"] == 100


def test_selector_applies_market_regime_sector_rps_thresholds():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame(
        [
            _selector_row(stock_code="600001.SH", market_regime="strong", sector_rps_5=56, sector_rps_10=51),
            _selector_row(stock_code="600002.SH", market_regime="weak", sector_rps_5=56, sector_rps_10=51),
        ]
    )

    selected = selector.select(rows, trade_date="2026-06-04")

    assert selected["watch_pool"]["stock_code"].tolist() == ["600001.SH"]


def test_selector_raises_when_sector_rps_columns_missing():
    selector = StockSelector(_selector_config())
    rows = pd.DataFrame([_selector_row()]).drop(columns=["sector_rps_5"])

    with pytest.raises(DataQualityError, match="sector_rps_5"):
        selector.select(rows, trade_date="2026-06-04")


def test_sector_factor_returns_neutral_when_sector_daily_missing(caplog):
    bars, sector_map, _ = _sector_fixture()
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    with caplog.at_level(logging.WARNING):
        result = SectorFactor(_sector_config()).calculate(
            bars,
            trade_date=trade_date,
            sector_map=sector_map,
            sector_daily=pd.DataFrame(),
        )

    assert set(result["score"]) == {50.0}
    assert set(result["sector_rps_5"]) == {50.0}
    assert set(result["sector_rps_pattern"]) == {"unknown"}
    assert set(result["active_sector_rps"]) == {50.0}
    assert "sector_daily missing" in caplog.text
