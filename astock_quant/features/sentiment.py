"""Market sentiment factor."""

from __future__ import annotations

import pandas as pd

from astock_quant.utils.calendar import ensure_no_future_data


class SentimentFactor:
    """Classify broad market regime and broadcast the sentiment score to stocks."""

    def calculate(self, daily_bars: pd.DataFrame, *, trade_date: str, limit_status: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return sentiment score rows for all stocks in the signal-day snapshot."""

        ensure_no_future_data(daily_bars, trade_date)
        if limit_status is not None:
            ensure_no_future_data(limit_status, trade_date)
        if daily_bars.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        bars = daily_bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        bars = bars.sort_values(["stock_code", "trade_date"])
        if "prev_close" not in bars.columns:
            bars["prev_close"] = bars.groupby("stock_code")["close"].shift(1)
        signal_date = pd.Timestamp(trade_date).normalize()
        snapshot = bars[bars["trade_date"] == signal_date].copy()
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        prev_close = snapshot["prev_close"]
        market_up_ratio = float((snapshot["close"] > prev_close.fillna(snapshot["close"])).mean())
        market_turnover_amount = float(snapshot.get("turnover_amount", pd.Series(0, index=snapshot.index)).sum())
        limit_up_count = int(snapshot.get("is_limit_up", pd.Series(False, index=snapshot.index)).fillna(False).sum())
        limit_down_count = int(snapshot.get("is_limit_down", pd.Series(False, index=snapshot.index)).fillna(False).sum())
        if limit_status is not None and not limit_status.empty:
            limit_up_count = int(limit_status.get("is_limit_up", pd.Series(dtype=bool)).fillna(False).sum())
            limit_down_count = int(limit_status.get("is_limit_down", pd.Series(dtype=bool)).fillna(False).sum())

        if market_up_ratio >= 0.65 and limit_up_count >= limit_down_count:
            regime = "strong"
            score = 85.0
        elif market_up_ratio < 0.35 and limit_down_count > limit_up_count:
            regime = "risk_off"
            score = 20.0
        elif market_up_ratio < 0.40:
            regime = "weak"
            score = 35.0
        else:
            regime = "neutral"
            score = 60.0

        result = snapshot[["stock_code", "trade_date"]].copy()
        result["score"] = score
        result["limit_up_count"] = limit_up_count
        result["limit_down_count"] = limit_down_count
        result["market_up_ratio"] = market_up_ratio
        result["market_turnover_amount"] = market_turnover_amount
        result["market_regime"] = regime
        return result.reset_index(drop=True)
