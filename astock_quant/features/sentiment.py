"""Market sentiment factor."""

from __future__ import annotations

from typing import Any

import pandas as pd

from astock_quant.utils.calendar import ensure_no_future_data


class SentimentFactor:
    """Classify broad market regime and broadcast the sentiment score to stocks.

    Single-day breadth is noisy, and the regime drives the total-position valve,
    so the up-ratio is smoothed over ``confirm_days`` and ``risk_off`` requires an
    absolute panic threshold of limit-down counts, not just a relative comparison.
    When index bars are provided, ``strong`` additionally requires the benchmark
    index to close above its 20-day moving average.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.confirm_days = max(int(self.config.get("confirm_days", 1) or 1), 1)
        self.risk_off_min_limit_down = int(self.config.get("risk_off_min_limit_down", 30))
        self.index_code = str(self.config.get("index_code", "000300.SH"))

    def calculate(
        self,
        daily_bars: pd.DataFrame,
        *,
        trade_date: str,
        limit_status: pd.DataFrame | None = None,
        index_bars: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return sentiment score rows for all stocks in the signal-day snapshot."""

        ensure_no_future_data(daily_bars, trade_date)
        if limit_status is not None:
            ensure_no_future_data(limit_status, trade_date)
        if index_bars is not None:
            ensure_no_future_data(index_bars, trade_date)
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

        market_up_ratio = self._up_ratio(snapshot)
        market_up_ratio_smoothed = self._smoothed_up_ratio(bars, signal_date, market_up_ratio)
        market_turnover_amount = float(snapshot.get("turnover_amount", pd.Series(0, index=snapshot.index)).sum())
        limit_up_count = int(snapshot.get("is_limit_up", pd.Series(False, index=snapshot.index)).fillna(False).sum())
        limit_down_count = int(snapshot.get("is_limit_down", pd.Series(False, index=snapshot.index)).fillna(False).sum())
        if limit_status is not None and not limit_status.empty:
            limit_up_count = int(limit_status.get("is_limit_up", pd.Series(dtype=bool)).fillna(False).sum())
            limit_down_count = int(limit_status.get("is_limit_down", pd.Series(dtype=bool)).fillna(False).sum())
        index_above_ma20 = self._index_above_ma20(index_bars, signal_date)

        if (
            market_up_ratio_smoothed < 0.35
            and limit_down_count > limit_up_count
            and limit_down_count >= self.risk_off_min_limit_down
        ):
            regime = "risk_off"
            score = 20.0
        elif market_up_ratio_smoothed < 0.40:
            regime = "weak"
            score = 35.0
        elif (
            market_up_ratio_smoothed >= 0.65
            and limit_up_count >= limit_down_count
            and index_above_ma20 is not False
        ):
            regime = "strong"
            score = 85.0
        else:
            regime = "neutral"
            score = 60.0

        result = snapshot[["stock_code", "trade_date"]].copy()
        result["score"] = score
        result["limit_up_count"] = limit_up_count
        result["limit_down_count"] = limit_down_count
        result["market_up_ratio"] = market_up_ratio
        result["market_up_ratio_smoothed"] = market_up_ratio_smoothed
        result["index_above_ma20"] = index_above_ma20
        result["market_turnover_amount"] = market_turnover_amount
        result["market_regime"] = regime
        return result.reset_index(drop=True)

    @staticmethod
    def _up_ratio(snapshot: pd.DataFrame) -> float:
        prev_close = snapshot["prev_close"]
        return float((snapshot["close"] > prev_close.fillna(snapshot["close"])).mean())

    def _smoothed_up_ratio(self, bars: pd.DataFrame, signal_date: pd.Timestamp, today_ratio: float) -> float:
        if self.confirm_days <= 1:
            return today_ratio
        recent_dates = (
            bars.loc[bars["trade_date"] <= signal_date, "trade_date"]
            .drop_duplicates()
            .sort_values()
            .tail(self.confirm_days)
        )
        ratios = []
        for date in recent_dates:
            day = bars[bars["trade_date"] == date]
            if not day.empty:
                ratios.append(self._up_ratio(day))
        if not ratios:
            return today_ratio
        return float(pd.Series(ratios).mean())

    def _index_above_ma20(self, index_bars: pd.DataFrame | None, signal_date: pd.Timestamp) -> bool | None:
        """Return None when no usable index data exists, so the filter stays inactive."""

        if index_bars is None or index_bars.empty or "close" not in index_bars.columns:
            return None
        idx = index_bars.copy()
        idx["trade_date"] = pd.to_datetime(idx["trade_date"]).dt.normalize()
        idx = idx[idx["trade_date"] <= signal_date]
        if idx.empty:
            return None
        if "index_code" in idx.columns:
            codes = set(idx["index_code"].dropna())
            chosen = self.index_code if self.index_code in codes else (sorted(codes)[0] if codes else None)
            if chosen is None:
                return None
            idx = idx[idx["index_code"] == chosen]
        idx = idx.sort_values("trade_date")
        closes = pd.to_numeric(idx["close"], errors="coerce").dropna()
        if len(closes) < 20:
            return None
        ma20 = float(closes.tail(20).mean())
        return bool(float(closes.iloc[-1]) > ma20)
