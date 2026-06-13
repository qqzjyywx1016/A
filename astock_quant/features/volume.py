"""Volume and liquidity confirmation factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.features.momentum import _rank_score
from astock_quant.utils.calendar import ensure_no_future_data


def _ramp(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    """Linear 0->1 ramp between low and high; smooth replacement for hard thresholds."""

    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if high <= low:
        return (array >= low).astype(float)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def long_upper_shadow_flag(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Long upper shadow measured from the candle body top, not from close.

    Using ``high - close`` mislabels big bearish candles as upper shadows because
    it includes the body; the shadow is ``high - max(open, close)``.
    """

    body_top = pd.concat([open_price, close], axis=1).max(axis=1)
    upper_shadow = high - body_top
    candle_range = (high - low).replace(0, np.nan)
    return (
        (upper_shadow / candle_range > 0.45)
        & (upper_shadow / close.replace(0, np.nan) > 0.03)
    ).fillna(False)


class VolumeFactor:
    """Score turnover strength and volume-price confirmation.

    The score is a continuous additive model: a tiny change in volume ratio or
    daily return moves the score gradually instead of jumping across hard branch
    boundaries, which keeps the factor robust to threshold noise.
    """

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
        snapshot["long_upper_shadow"] = long_upper_shadow_flag(
            snapshot["open"], snapshot["high"], snapshot["low"], snapshot["close"]
        )
        snapshot["score"] = self._smooth_score(snapshot)
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

    @staticmethod
    def _smooth_score(snapshot: pd.DataFrame) -> pd.Series:
        volume_ratio = pd.to_numeric(snapshot["volume_ratio_20d"], errors="coerce")
        return_1d = pd.to_numeric(snapshot["return_1d"], errors="coerce").fillna(0.0)
        close = pd.to_numeric(snapshot["close"], errors="coerce")
        open_price = pd.to_numeric(snapshot["open"], errors="coerce")
        high = pd.to_numeric(snapshot["high"], errors="coerce")
        low = pd.to_numeric(snapshot["low"], errors="coerce")
        ma10 = pd.to_numeric(snapshot["ma10"], errors="coerce")

        # Intensity of the up/down move, saturating at +-1% (down) and +2% (up).
        up_move = _ramp(return_1d, 0.0, 0.02)
        down_move = _ramp(-return_1d, 0.0, 0.01)
        # Volume regimes: expansion saturates at 1.5x, blow-off builds from 2.5x to 3.5x.
        expansion = _ramp(volume_ratio, 0.8, 1.5)
        blowoff = _ramp(volume_ratio, 2.5, 3.5)
        shrink = 1.0 - _ramp(volume_ratio, 0.7, 0.95)
        above_ma10 = (close > ma10).fillna(False).to_numpy(dtype=float)

        body_top = pd.concat([open_price, close], axis=1).max(axis=1)
        candle_range = (high - low).replace(0, np.nan)
        shadow_ratio = ((high - body_top) / candle_range).fillna(0.0)
        shadow_pct = ((high - body_top) / close.replace(0, np.nan)).fillna(0.0)
        shadow_severity = _ramp(shadow_ratio, 0.30, 0.55) * _ramp(shadow_pct, 0.015, 0.04)
        bearish_candle = (close < open_price).fillna(False).to_numpy(dtype=float)

        score = np.full(len(snapshot), 50.0)
        # Healthy expansion with an up move is the core confirmation pattern.
        score += 40.0 * up_move * expansion * (1.0 - blowoff)
        # Shrinking volume: gentle pullback above MA10 is constructive, drift-up is mildly positive.
        score += 30.0 * down_move * shrink * above_ma10
        score += 20.0 * up_move * shrink
        # Risk patterns: blow-off surge, heavy-volume decline, stagnation, long upper shadow.
        score -= 30.0 * blowoff * up_move
        score -= 35.0 * down_move * expansion * bearish_candle
        score -= 25.0 * expansion * (1.0 - up_move) * (1.0 - down_move)
        score -= 35.0 * shadow_severity * _ramp(volume_ratio, 1.5, 3.0)

        return pd.Series(score, index=snapshot.index).clip(0, 100).round(2)
