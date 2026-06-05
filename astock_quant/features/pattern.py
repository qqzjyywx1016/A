"""Price-pattern factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.utils.calendar import ensure_no_future_data


class PatternFactor:
    """Score moving-average alignment and flag exhaustion patterns."""

    def calculate(self, daily_bars: pd.DataFrame, *, trade_date: str) -> pd.DataFrame:
        """Return one pattern score row per stock for the signal date."""

        ensure_no_future_data(daily_bars, trade_date)
        if daily_bars.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        bars = daily_bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        signal_date = pd.Timestamp(trade_date).normalize()
        bars = bars.sort_values(["stock_code", "trade_date"])
        grouped = bars.groupby("stock_code", group_keys=False)
        for window in [5, 10, 20]:
            bars[f"ma{window}"] = grouped["close"].transform(lambda s: s.rolling(window, min_periods=1).mean())
        bars["rolling_20d_high_prev"] = grouped["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).max())
        bars["avg_turnover_amount_20d"] = grouped["turnover_amount"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        bars["prev_close_calc"] = grouped["close"].shift(1)
        snapshot = bars[bars["trade_date"] == signal_date].copy()
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        snapshot["above_ma5"] = snapshot["close"] > snapshot["ma5"]
        snapshot["above_ma10"] = snapshot["close"] > snapshot["ma10"]
        snapshot["above_ma20"] = snapshot["close"] > snapshot["ma20"]
        snapshot["breakout_20d_high"] = snapshot["close"] >= snapshot["rolling_20d_high_prev"].fillna(snapshot["high"])
        snapshot["close_near_high"] = (snapshot["high"] - snapshot["close"]) / snapshot["close"].replace(0, np.nan) <= 0.02
        candle_range = (snapshot["high"] - snapshot["low"]).replace(0, np.nan)
        snapshot["long_upper_shadow"] = (
            ((snapshot["high"] - snapshot["close"]) / candle_range > 0.45)
            & ((snapshot["high"] - snapshot["close"]) / snapshot["close"].replace(0, np.nan) > 0.03)
        ).fillna(False)
        daily_return = snapshot["close"] / snapshot["prev_close_calc"].replace(0, np.nan) - 1
        volume_ratio = snapshot["turnover_amount"] / snapshot["avg_turnover_amount_20d"].replace(0, np.nan)
        snapshot["high_volume_stagnation"] = ((volume_ratio > 1.5) & (daily_return.fillna(0) < 0.01)).fillna(False)
        snapshot["high_volume_bearish"] = (
            (volume_ratio > 1.5)
            & (snapshot["close"] < snapshot["open"])
            & (daily_return.fillna(0) <= -0.02)
        ).fillna(False)
        positive = (
            snapshot["above_ma5"].astype(int)
            + snapshot["above_ma10"].astype(int)
            + snapshot["above_ma20"].astype(int)
            + snapshot["breakout_20d_high"].astype(int)
            + snapshot["close_near_high"].astype(int)
        ) * 18
        penalty = snapshot["long_upper_shadow"].astype(int) * 20 + snapshot["high_volume_stagnation"].astype(int) * 20
        snapshot["score"] = (positive - penalty + 10).clip(0, 100).round(2)
        columns = [
            "stock_code",
            "trade_date",
            "score",
            "above_ma5",
            "above_ma10",
            "above_ma20",
            "breakout_20d_high",
            "close_near_high",
            "long_upper_shadow",
            "high_volume_stagnation",
            "high_volume_bearish",
        ]
        return snapshot[columns].reset_index(drop=True)
