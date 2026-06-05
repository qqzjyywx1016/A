import pandas as pd

from astock_quant.universe.filters import UniverseFilter


def test_universe_filter_excludes_st_suspended_low_liquidity_and_market_cap():
    config = {
        "exclude_st": True,
        "exclude_suspended": True,
        "exclude_bj": True,
        "min_listing_days": 60,
        "min_turnover_amount": 200_000_000,
        "min_avg_turnover_amount_20d": 100_000_000,
        "min_float_market_cap": 2_000_000_000,
        "max_float_market_cap": 30_000_000_000,
    }
    stocks = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "stock_name": "强势股份",
                "is_st": False,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 150_000_000,
                "float_market_cap": 10_000_000_000,
            },
            {
                "stock_code": "600002.SH",
                "stock_name": "*ST 风险",
                "is_st": True,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 150_000_000,
                "float_market_cap": 10_000_000_000,
            },
            {
                "stock_code": "600003.SH",
                "stock_name": "停牌股份",
                "is_st": False,
                "is_suspended": True,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 150_000_000,
                "float_market_cap": 10_000_000_000,
            },
            {
                "stock_code": "430001.BJ",
                "stock_name": "北交股份",
                "is_st": False,
                "is_suspended": False,
                "exchange": "BJ",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 150_000_000,
                "float_market_cap": 10_000_000_000,
            },
            {
                "stock_code": "600004.SH",
                "stock_name": "低成交额",
                "is_st": False,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 100_000_000,
                "avg_turnover_amount_20d": 150_000_000,
                "float_market_cap": 10_000_000_000,
            },
            {
                "stock_code": "600005.SH",
                "stock_name": "超大市值",
                "is_st": False,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 150_000_000,
                "float_market_cap": 50_000_000_000,
            },
        ]
    )

    result = UniverseFilter(config).apply(stocks)

    assert result["stock_code"].tolist() == ["600001.SH"]


def test_universe_filter_raises_when_input_uses_future_dates():
    config = {"min_turnover_amount": 0}
    stocks = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "turnover_amount": 1},
            {"stock_code": "600002.SH", "trade_date": "2026-06-05", "turnover_amount": 1},
        ]
    )

    try:
        UniverseFilter(config).apply(stocks, as_of_date="2026-06-04")
    except ValueError as exc:
        assert "future" in str(exc).lower()
    else:
        raise AssertionError("future-dated rows must be rejected")


def test_universe_filter_uses_exchange_for_bj_and_does_not_drop_kechuang_by_prefix():
    config = {
        "exclude_bj": True,
        "min_turnover_amount": 0,
    }
    stocks = pd.DataFrame(
        [
            {"stock_code": "688001.SH", "stock_name": "科创股份", "exchange": "SH", "turnover_amount": 1},
            {"stock_code": "430001.SZ", "stock_name": "非北交兜底", "exchange": "SZ", "turnover_amount": 1},
            {"stock_code": "830001.BJ", "stock_name": "北交股份", "exchange": "BJ", "turnover_amount": 1},
            {"stock_code": "870001.BJ", "stock_name": "北交股份二", "exchange": "BJ", "turnover_amount": 1},
        ]
    )

    result = UniverseFilter(config).apply(stocks)

    assert result["stock_code"].tolist() == ["688001.SH", "430001.SZ"]


def test_universe_filter_excludes_event_risk_when_columns_exist():
    config = {
        "min_turnover_amount": 0,
        "earnings_blackout_days": 3,
        "lockup_blackout_days": 5,
    }
    stocks = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "turnover_amount": 1, "next_report_date": "2026-06-06"},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "turnover_amount": 1, "lockup_date": "2026-06-09"},
            {"stock_code": "600003.SH", "trade_date": "2026-06-04", "turnover_amount": 1, "is_restructuring": True},
            {"stock_code": "600004.SH", "trade_date": "2026-06-04", "turnover_amount": 1, "is_major_event": True},
            {"stock_code": "600005.SH", "trade_date": "2026-06-04", "turnover_amount": 1, "next_report_date": "2026-06-20"},
        ]
    )

    result = UniverseFilter(config).apply(stocks, as_of_date="2026-06-04")

    assert result["stock_code"].tolist() == ["600005.SH"]


def test_universe_filter_event_hooks_noop_when_columns_missing():
    config = {
        "min_turnover_amount": 0,
        "earnings_blackout_days": 3,
        "lockup_blackout_days": 5,
    }
    stocks = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "turnover_amount": 1},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "turnover_amount": 1},
        ]
    )

    result = UniverseFilter(config).apply(stocks, as_of_date="2026-06-04")

    assert result["stock_code"].tolist() == ["600001.SH", "600002.SH"]


def test_universe_filter_default_stage_one_market_cap_floor_without_upper_cap():
    config = {
        "exclude_st": True,
        "exclude_suspended": True,
        "exclude_bj": True,
        "min_listing_days": 60,
        "min_turnover_amount": 200_000_000,
        "min_avg_turnover_amount_20d": 80_000_000,
        "min_avg_turnover_rate_20d": 0.01,
        "min_float_market_cap": 3_000_000_000,
        "max_float_market_cap": None,
    }
    stocks = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "stock_name": "小盘达标",
                "is_st": False,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 100_000_000,
                "avg_turnover_rate_20d": 0.02,
                "float_market_cap": 3_000_000_000,
            },
            {
                "stock_code": "600002.SH",
                "stock_name": "低市值",
                "is_st": False,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 100_000_000,
                "avg_turnover_rate_20d": 0.02,
                "float_market_cap": 2_999_999_999,
            },
            {
                "stock_code": "600003.SH",
                "stock_name": "超大市值保留",
                "is_st": False,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 100_000_000,
                "avg_turnover_rate_20d": 0.02,
                "float_market_cap": 80_000_000_000,
            },
            {
                "stock_code": "600004.SH",
                "stock_name": "低换手",
                "is_st": False,
                "is_suspended": False,
                "exchange": "SH",
                "listing_days": 120,
                "turnover_amount": 300_000_000,
                "avg_turnover_amount_20d": 100_000_000,
                "avg_turnover_rate_20d": 0.009,
                "float_market_cap": 4_000_000_000,
            },
        ]
    )

    result = UniverseFilter(config).apply(stocks)

    assert result["stock_code"].tolist() == ["600001.SH", "600003.SH"]


def test_universe_filter_skips_avg_turnover_rate_when_column_missing():
    config = {
        "min_turnover_amount": 0,
        "min_avg_turnover_rate_20d": 0.01,
        "min_float_market_cap": 3_000_000_000,
        "max_float_market_cap": None,
    }
    stocks = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "turnover_amount": 1, "float_market_cap": 3_000_000_000},
            {"stock_code": "600002.SH", "turnover_amount": 1, "float_market_cap": 4_000_000_000},
        ]
    )

    result = UniverseFilter(config).apply(stocks)

    assert result["stock_code"].tolist() == ["600001.SH", "600002.SH"]
