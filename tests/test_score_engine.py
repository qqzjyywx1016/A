import pandas as pd

from astock_quant.scoring.score_engine import ScoreEngine


def test_score_engine_applies_weights_and_rating_thresholds():
    weights = {
        "momentum": 0.25,
        "volume": 0.20,
        "sector": 0.20,
        "fund_flow": 0.15,
        "pattern": 0.10,
        "sentiment": 0.10,
    }
    factors = {
        "momentum": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 100}]
        ),
        "volume": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]
        ),
        "sector": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 70}]
        ),
        "fund_flow": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 60}]
        ),
        "pattern": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 90}]
        ),
        "sentiment": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 55}]
        ),
    }
    stock_basic = pd.DataFrame(
        [{"stock_code": "600001.SH", "stock_name": "强势股份", "sector": "机器人"}]
    )

    scored = ScoreEngine(weights).score(factors, stock_basic)

    row = scored.iloc[0]
    assert row["total_score"] == 78.5
    assert row["momentum_score"] == 100
    assert row["volume_score"] == 80
    assert row["sector_score"] == 70
    assert row["fund_score"] == 60
    assert row["pattern_score"] == 90
    assert row["sentiment_score"] == 55
    assert row["rating"] == "B"


def test_score_engine_fills_missing_fund_flow_with_neutral_score():
    weights = {
        "momentum": 0.25,
        "volume": 0.20,
        "sector": 0.20,
        "fund_flow": 0.15,
        "pattern": 0.10,
        "sentiment": 0.10,
    }
    factors = {
        "momentum": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]
        ),
        "volume": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]
        ),
        "sector": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]
        ),
        "pattern": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]
        ),
        "sentiment": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]
        ),
    }

    scored = ScoreEngine(weights).score(factors)

    assert scored.iloc[0]["fund_score"] == 50
    assert scored.iloc[0]["total_score"] == 80.0
    assert scored.iloc[0]["rating"] == "A"


def test_score_engine_passes_through_rps_fields_without_coercing_labels():
    weights = {
        "momentum": 1.0,
        "volume": 0.0,
        "sector": 0.0,
        "fund_flow": 0.0,
        "pattern": 0.0,
        "sentiment": 0.0,
    }
    factors = {
        "momentum": pd.DataFrame(
            [
                {
                    "stock_code": "600001.SH",
                    "trade_date": "2026-06-04",
                    "score": 88,
                    "rps_5": "91",
                    "rps_10": 86,
                    "rps_20": 80,
                    "rps_60": 70,
                    "rps_composite": 84.2,
                    "rps_pattern": "acceleration",
                    "return_5d": 0.08,
                    "return_10d": 0.12,
                    "above_ma20": True,
                    "ma20": 10.5,
                    "is_20d_high": True,
                    "is_60d_high": False,
                }
            ]
        )
    }

    scored = ScoreEngine(weights).score(factors)
    row = scored.iloc[0]

    assert row["rps_5"] == 91
    assert row["rps_pattern"] == "acceleration"
    assert row["above_ma20"] is True


def test_score_engine_warns_when_factor_is_neutral(caplog):
    weights = {
        "momentum": 0.25,
        "volume": 0.20,
        "sector": 0.20,
        "fund_flow": 0.15,
        "pattern": 0.10,
        "sentiment": 0.10,
    }
    factors = {
        "momentum": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]
        ),
        "fund_flow": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 50}]
        ),
    }

    ScoreEngine(weights).score(factors)

    assert "fund_flow" in caplog.text


def test_score_engine_renormalizes_weights_when_factor_is_all_neutral():
    weights = {
        "momentum": 0.25,
        "volume": 0.20,
        "sector": 0.20,
        "fund_flow": 0.15,
        "pattern": 0.10,
        "sentiment": 0.10,
    }
    factors = {
        "momentum": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 100}]),
        "volume": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 80}]),
        "sector": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 50}]),
        "fund_flow": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 60}]),
        "pattern": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 90}]),
        "sentiment": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 70}]),
    }

    scored = ScoreEngine(weights).score(factors)

    expected = round((100 * 0.25 + 80 * 0.20 + 60 * 0.15 + 90 * 0.10 + 70 * 0.10) / 0.80, 2)
    assert scored.iloc[0]["total_score"] == expected


def test_score_engine_sets_missing_boolean_passthrough_to_false():
    weights = {"momentum": 1.0}
    factors = {
        "momentum": pd.DataFrame(
            [
                {
                    "stock_code": "600001.SH",
                    "trade_date": "2026-06-04",
                    "score": 88,
                    "above_ma20": pd.NA,
                    "is_20d_high": pd.NA,
                    "is_60d_high": pd.NA,
                }
            ]
        )
    }

    scored = ScoreEngine(weights).score(factors)

    assert scored.iloc[0]["above_ma20"] is False
    assert scored.iloc[0]["is_20d_high"] is False
    assert scored.iloc[0]["is_60d_high"] is False


def test_score_engine_excludes_zero_weight_sentiment_but_passes_market_regime():
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
        "reversal": pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 50}]),
        "sentiment": pd.DataFrame(
            [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "score": 10, "market_regime": "risk_off"}]
        ),
    }

    scored = ScoreEngine(weights).score(factors)

    expected = round((80 * 0.25 + 70 * 0.20 + 60 * 0.15 + 90 * 0.15) / 0.75, 2)
    assert scored.iloc[0]["total_score"] == expected
    assert scored.iloc[0]["sentiment_score"] == 10
    assert scored.iloc[0]["market_regime"] == "risk_off"
