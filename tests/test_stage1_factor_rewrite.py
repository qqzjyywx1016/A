import math

import pandas as pd

from astock_quant.features.momentum import MomentumFactor
from astock_quant.features.reversal import ReversalFactor
from astock_quant.features.volume import VolumeFactor
from astock_quant.scoring.score_engine import ScoreEngine


def _panel(stock_count: int = 25, periods: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2026-02-27", periods=periods)
    rows = []
    for stock_index in range(stock_count):
        code = f"{stock_index + 1:06d}.SZ"
        drift = 0.01 + stock_index * 0.0005
        close = 20 + stock_index
        for day_index, trade_date in enumerate(dates):
            close *= 1 + drift + (0.001 if day_index % 3 == 0 else -0.0005)
            rows.append(
                {
                    "stock_code": code,
                    "trade_date": trade_date,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "turnover_amount": 100_000_000 + stock_index * 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_momentum_score_uses_rps20_rps60_trend_efficiency_not_rps5_rps10():
    bars = _panel()
    trade_date = bars["trade_date"].max().strftime("%Y-%m-%d")

    result = MomentumFactor().calculate(bars, trade_date=trade_date)
    row = result[result["stock_code"] == "000020.SZ"].iloc[0]

    expected = round(
        min(
            max(
                0.35 * row["rps_20"]
                + 0.25 * row["rps_60"]
                + 0.25 * row["trend_efficiency_score"]
                + 0.15 * (100 if row["is_60d_high"] else 0),
                0,
            ),
            100,
        ),
        2,
    )
    assert row["score"] == expected


def test_trend_efficiency_is_finite_when_volatility_is_zero():
    dates = pd.bdate_range("2026-02-27", periods=70)
    rows = []
    for stock_index in range(25):
        close = 10 + stock_index
        for trade_date in dates:
            close += 1
            rows.append(
                {
                    "stock_code": f"{stock_index + 1:06d}.SZ",
                    "trade_date": trade_date,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "turnover_amount": 100_000_000,
                }
            )

    result = MomentumFactor().calculate(pd.DataFrame(rows), trade_date=dates[-1].strftime("%Y-%m-%d"))

    assert result["trend_efficiency"].map(math.isfinite).all()
    assert result["score"].between(0, 100).all()


def test_reversal_factor_scores_orderly_pullback_above_deep_or_no_pullback():
    dates = pd.bdate_range("2026-05-25", periods=8)
    rows = []
    closes = {
        "600001.SH": [10, 10.2, 10.5, 10.8, 11.0, 10.7, 10.55, 10.56],
        "600002.SH": [10, 10.2, 10.4, 10.6, 10.8, 10.7, 10.7, 10.8],
        "600003.SH": [10, 10.5, 11.0, 11.2, 11.5, 10.5, 9.6, 9.4],
    }
    amounts = {"600001.SH": 50, "600002.SH": 100, "600003.SH": 100}
    for code, series in closes.items():
        for trade_date, close in zip(dates, series, strict=True):
            rows.append(
                {
                    "stock_code": code,
                    "trade_date": trade_date,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "turnover_amount": amounts[code],
                }
            )

    result = ReversalFactor().calculate(pd.DataFrame(rows), trade_date=dates[-1].strftime("%Y-%m-%d"))

    by_code = result.set_index("stock_code")
    assert by_code.loc["600001.SH", "reversal_score"] > by_code.loc["600002.SH", "reversal_score"]
    assert by_code.loc["600001.SH", "reversal_score"] > by_code.loc["600003.SH", "reversal_score"]
    assert {"pullback_score", "decel_score", "shrink_score"}.issubset(result.columns)


def _volume_case_rows(code: str, current: dict) -> list[dict]:
    dates = pd.bdate_range("2026-05-07", periods=21)
    rows = []
    for trade_date in dates[:-1]:
        rows.append(
            {
                "stock_code": code,
                "trade_date": trade_date,
                "open": 98,
                "high": 100,
                "low": 97,
                "close": 98,
                "turnover_amount": 100,
            }
        )
    rows[-1]["close"] = 100
    rows[-1]["open"] = 100
    rows.append({"stock_code": code, "trade_date": dates[-1], **current})
    return rows


def test_volume_factor_decision_tree_branches():
    trade_date = "2026-06-04"
    cases = {
        "600001.SH": {"open": 100, "high": 120, "low": 99, "close": 107, "turnover_amount": 500},
        "600002.SH": {"open": 100, "high": 101, "low": 95, "close": 96, "turnover_amount": 400},
        "600003.SH": {"open": 100, "high": 101, "low": 99, "close": 100.5, "turnover_amount": 200},
        "600004.SH": {"open": 100, "high": 103, "low": 99, "close": 102, "turnover_amount": 150},
        "600005.SH": {"open": 100, "high": 100.5, "low": 98, "close": 99, "turnover_amount": 70},
        "600006.SH": {"open": 100, "high": 102, "low": 99, "close": 101, "turnover_amount": 70},
        "600007.SH": {"open": 100, "high": 101, "low": 99, "close": 100, "turnover_amount": 100},
    }
    rows = []
    for code, current in cases.items():
        rows.extend(_volume_case_rows(code, current))

    result = VolumeFactor().calculate(pd.DataFrame(rows), trade_date=trade_date).set_index("stock_code")

    assert result.loc["600001.SH", "score"] == 10
    assert result.loc["600002.SH", "score"] == 10
    assert result.loc["600003.SH", "score"] == 25
    assert result.loc["600004.SH", "score"] == 90
    assert result.loc["600005.SH", "score"] == 80
    assert result.loc["600006.SH", "score"] == 60
    assert result.loc["600007.SH", "score"] == 50


def test_score_engine_includes_reversal_and_market_cap_scores_in_total():
    weights = {
        "momentum": 0.25,
        "volume": 0.20,
        "sector": 0.15,
        "market_cap": 0.15,
        "reversal": 0.15,
        "fund_flow": 0.0,
        "sentiment": 0.0,
    }
    factors = {
        "momentum": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]),
        "volume": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 70}]),
        "sector": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 60}]),
        "market_cap": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 90}]),
        "reversal": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 85}]),
        "sentiment": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 10, "market_regime": "weak"}]
        ),
    }

    scored = ScoreEngine(weights).score(factors)

    expected = round((80 * 0.25 + 70 * 0.20 + 60 * 0.15 + 90 * 0.15 + 85 * 0.15) / 0.90, 2)
    assert scored.iloc[0]["total_score"] == expected
    assert scored.iloc[0]["market_cap_score"] == 90
    assert scored.iloc[0]["reversal_score"] == 85
