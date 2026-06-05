import pandas as pd

from astock_quant.features.sentiment import SentimentFactor


def test_sentiment_uses_grouped_previous_close_before_signal_snapshot():
    bars = pd.DataFrame(
        [
            {"stock_code": "000001.SZ", "trade_date": "2026-06-03", "close": 10, "turnover_amount": 100},
            {"stock_code": "000002.SZ", "trade_date": "2026-06-03", "close": 20, "turnover_amount": 100},
            {"stock_code": "000001.SZ", "trade_date": "2026-06-04", "close": 11, "turnover_amount": 100},
            {"stock_code": "000002.SZ", "trade_date": "2026-06-04", "close": 19, "turnover_amount": 100},
        ]
    )

    result = SentimentFactor().calculate(bars, trade_date="2026-06-04")

    assert result.iloc[0]["market_up_ratio"] == 0.5
    assert result.iloc[0]["market_regime"] == "neutral"
