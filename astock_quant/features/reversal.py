"""Short-term pullback reversal factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.utils.calendar import ensure_no_future_data


class ReversalFactor:
    """Score orderly pullbacks inside a still-tradable momentum setup."""

    def calculate(self, daily_bars: pd.DataFrame, *, trade_date: str) -> pd.DataFrame:
        """Return one short-term reversal score row per stock for the signal date."""

        ensure_no_future_data(daily_bars, trade_date)
        if daily_bars.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score", "reversal_score"])

        bars = daily_bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        signal_date = pd.Timestamp(trade_date).normalize()
        bars = bars.sort_values(["stock_code", "trade_date"])
        grouped = bars.groupby("stock_code", group_keys=False)
        bars["return_1d"] = grouped["close"].pct_change(1)
        bars["recent_high_5d"] = grouped["close"].transform(lambda s: s.rolling(5, min_periods=1).max())
        bars["pullback"] = (bars["recent_high_5d"] - bars["close"]) / bars["recent_high_5d"].replace(0, np.nan)
        bars["prev_return_4d_mean"] = grouped["return_1d"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        bars["avg_turnover_amount_10d"] = grouped["turnover_amount"].transform(
            lambda s: s.rolling(10, min_periods=1).mean()
        )

        snapshot = bars[bars["trade_date"] == signal_date].copy()
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score", "reversal_score"])

        snapshot["pullback_score"] = self._pullback_score(snapshot["pullback"])
        diff = snapshot["return_1d"] - snapshot["prev_return_4d_mean"].fillna(snapshot["return_1d"])
        snapshot["decel_score"] = np.where(diff > 0, 100.0, (100 + diff * 1000).clip(0, 100))
        snapshot["shrink_score"] = np.where(
            snapshot["turnover_amount"] < snapshot["avg_turnover_amount_10d"] * 0.7,
            100.0,
            0.0,
        )
        snapshot["reversal_score"] = (
            snapshot["pullback_score"] * 0.50 + snapshot["decel_score"] * 0.30 + snapshot["shrink_score"] * 0.20
        ).clip(0, 100).round(2)
        snapshot["score"] = snapshot["reversal_score"]
        columns = [
            "stock_code",
            "trade_date",
            "score",
            "reversal_score",
            "pullback_score",
            "decel_score",
            "shrink_score",
            "recent_high_5d",
            "pullback",
        ]
        return snapshot[columns].reset_index(drop=True)

    @staticmethod
    def _pullback_score(pullback: pd.Series) -> pd.Series:
        values = pd.to_numeric(pullback, errors="coerce").fillna(0)
        score = pd.Series(0.0, index=values.index)
        rising = (values >= 0.01) & (values <= 0.04)
        score.loc[rising] = ((values.loc[rising] - 0.01) / 0.03 * 100).clip(0, 100)
        mild = (values > 0.04) & (values <= 0.08)
        score.loc[mild] = (100 - (values.loc[mild] - 0.04) / 0.04 * 40).clip(60, 100)
        deep = (values > 0.08) & (values <= 0.15)
        score.loc[deep] = (60 * (0.15 - values.loc[deep]) / 0.07).clip(0, 60)
        return score
