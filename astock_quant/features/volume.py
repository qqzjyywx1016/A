"""Volume and liquidity confirmation factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.features.momentum import _rank_score
from astock_quant.utils.calendar import ensure_no_future_data


class VolumeFactor:
    """Score turnover strength and volume-price confirmation."""

    def calculate(self, daily_bars: pd.DataFrame, *, trade_date: str) -> pd.DataFrame:
        """Return one volume score row per stock for the signal date."""

        ensure_no_future_data(daily_bars, trade_date)
        if daily_bars.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        bars = daily_bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        signal_date = pd.Timestamp(trade_date).normalize()
        bars = bars.sort_values(["stock_code", "trade_date"])
        grouped = bars.groupby("stock_code", group_keys=False)
        bars["avg_turnover_amount_20d"] = grouped["turnover_amount"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        bars["amount_60d_max"] = grouped["turnover_amount"].transform(lambda s: s.rolling(60, min_periods=20).max())
        bars["volume_ratio_20d"] = bars["turnover_amount"] / bars["avg_turnover_amount_20d"].replace(0, np.nan)
        bars["ma10"] = grouped["close"].transform(lambda s: s.rolling(10, min_periods=1).mean())
        if "turnover_rate" not in bars.columns:
            bars["turnover_rate"] = np.nan
        bars["prev_close_calc"] = grouped["close"].shift(1)

        snapshot = bars[bars["trade_date"] == signal_date].copy()
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        snapshot["amount_rank_pct"] = _rank_score(snapshot["turnover_amount"]) / 100
        snapshot["is_volume_price_confirmed"] = (
            (snapshot["close"] > snapshot["prev_close_calc"].fillna(snapshot.get("prev_close", snapshot["close"])))
            & (snapshot["volume_ratio_20d"] >= 1.2)
        )
        snapshot["return_1d"] = snapshot["close"] / snapshot["prev_close_calc"].replace(0, pd.NA) - 1
        candle_range = (snapshot["high"] - snapshot["low"]).replace(0, pd.NA)
        snapshot["long_upper_shadow"] = (
            ((snapshot["high"] - snapshot["close"]) / candle_range > 0.45)
            & ((snapshot["high"] - snapshot["close"]) / snapshot["close"].replace(0, pd.NA) > 0.03)
        ).fillna(False)
        snapshot["score"] = 50.0
        conditions = [
            (snapshot["volume_ratio_20d"] > 3) & snapshot["long_upper_shadow"],
            (snapshot["volume_ratio_20d"] > 2.5) & (snapshot["close"] < snapshot["open"]) & (snapshot["return_1d"] < -0.03),
            (snapshot["volume_ratio_20d"] > 1.5) & (snapshot["return_1d"] < 0.01),
            (snapshot["volume_ratio_20d"].between(1.2, 2.5, inclusive="both")) & (snapshot["return_1d"] > 0),
            (snapshot["volume_ratio_20d"] < 0.8) & (snapshot["return_1d"] < 0) & (snapshot["close"] > snapshot["ma10"]),
            (snapshot["volume_ratio_20d"] < 0.8) & (snapshot["return_1d"] > 0),
        ]
        scores = [10.0, 10.0, 25.0, 90.0, 80.0, 60.0]
        remaining = pd.Series(True, index=snapshot.index)
        for condition, score in zip(conditions, scores, strict=True):
            hits = remaining & condition.fillna(False)
            snapshot.loc[hits, "score"] = score
            remaining &= ~hits
        snapshot["score"] = snapshot["score"].clip(0, 100).round(2)
        columns = [
            "stock_code",
            "trade_date",
            "score",
            "turnover_amount",
            "amount_60d_max",
            "avg_turnover_amount_20d",
            "volume_ratio_20d",
            "turnover_rate",
            "amount_rank_pct",
            "is_volume_price_confirmed",
            "return_1d",
            "ma10",
            "long_upper_shadow",
        ]
        return snapshot[columns].reset_index(drop=True)
