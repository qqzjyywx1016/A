import pandas as pd

from astock_quant.features.market_cap import MarketCapFactor


def test_market_cap_factor_scores_small_mid_large_tiers():
    snapshot = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "trade_date": "2026-06-04",
                "float_market_cap": 5_000_000_000,
                "avg_turnover_amount_20d": 300_000_000,
                "avg_turnover_rate_20d": 0.04,
                "active_sector_regime": "strong",
            },
            {
                "stock_code": "600002.SH",
                "trade_date": "2026-06-04",
                "float_market_cap": 12_000_000_000,
                "avg_turnover_amount_20d": 200_000_000,
                "avg_turnover_rate_20d": 0.03,
                "active_sector_regime": "neutral",
            },
            {
                "stock_code": "600003.SH",
                "trade_date": "2026-06-04",
                "float_market_cap": 30_000_000_000,
                "avg_turnover_amount_20d": 100_000_000,
                "avg_turnover_rate_20d": 0.02,
                "active_sector_regime": "neutral",
            },
            {
                "stock_code": "600004.SH",
                "trade_date": "2026-06-04",
                "float_market_cap": 60_000_000_000,
                "avg_turnover_amount_20d": 50_000_000,
                "avg_turnover_rate_20d": 0.01,
                "active_sector_regime": "neutral",
            },
        ]
    )

    result = MarketCapFactor().calculate(snapshot, trade_date="2026-06-04")

    by_code = result.set_index("stock_code")
    assert by_code.loc["600001.SH", "market_cap_score"] == 100.0
    assert by_code.loc["600002.SH", "market_cap_score"] == 73.5
    assert by_code.loc["600003.SH", "market_cap_score"] == 54.0
    assert by_code.loc["600004.SH", "market_cap_score"] == 37.5


def test_market_cap_factor_uses_sector_regime_map_for_bonus():
    snapshot = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "trade_date": "2026-06-04",
                "float_market_cap": 5_000_000_000,
                "avg_turnover_amount_20d": 100_000_000,
                "avg_turnover_rate_20d": 0.02,
                "active_sector_code": "BK001",
            },
            {
                "stock_code": "600002.SH",
                "trade_date": "2026-06-04",
                "float_market_cap": 12_000_000_000,
                "avg_turnover_amount_20d": 100_000_000,
                "avg_turnover_rate_20d": 0.02,
                "active_sector_code": "BK002",
            },
        ]
    )

    result = MarketCapFactor().calculate(
        snapshot,
        trade_date="2026-06-04",
        sector_regime_map={"BK001": "strong", "BK002": "neutral"},
    )

    by_code = result.set_index("stock_code")
    assert by_code.loc["600001.SH", "market_cap_score"] > by_code.loc["600002.SH", "market_cap_score"]
