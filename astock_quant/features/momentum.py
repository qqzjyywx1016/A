"""Momentum and relative-strength factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.utils.calendar import ensure_no_future_data
from astock_quant.utils.logger import get_logger


logger = get_logger(__name__)
MIN_CROSS_SECTION = 20
RPS_WEIGHTS = {"rps_5": 0.35, "rps_10": 0.30, "rps_20": 0.25, "rps_60": 0.10}


def _rank_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1 or values.nunique(dropna=True) <= 1:
        return pd.Series(np.where(values.notna(), 50.0, 50.0), index=series.index)
    return values.rank(pct=True).fillna(0.5) * 100


class MomentumFactor:
    """Calculate short-horizon momentum, relative return and new-high scores."""

    def calculate(
        self,
        daily_bars: pd.DataFrame,
        *,
        trade_date: str,
        index_bars: pd.DataFrame | None = None,
        sector_daily: pd.DataFrame | None = None,
        sector_map: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return one score row per stock for the signal date."""

        ensure_no_future_data(daily_bars, trade_date)
        if index_bars is not None:
            ensure_no_future_data(index_bars, trade_date)
        if sector_daily is not None:
            ensure_no_future_data(sector_daily, trade_date)

        if daily_bars.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        bars = daily_bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        signal_date = pd.Timestamp(trade_date).normalize()
        bars = bars.sort_values(["stock_code", "trade_date"])
        grouped = bars.groupby("stock_code", group_keys=False)
        for window in [1, 3, 5, 10, 20, 60]:
            bars[f"return_{window}d"] = grouped["close"].pct_change(window)
        bars["daily_return"] = grouped["close"].pct_change(1)
        bars["volatility_20d"] = grouped["daily_return"].transform(lambda s: s.rolling(20, min_periods=20).std(ddof=1))
        volatility_floor = pd.to_numeric(bars["volatility_20d"], errors="coerce").clip(lower=1e-3)
        bars["trend_efficiency"] = bars["return_20d"] / volatility_floor
        bars["ma20"] = grouped["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        bars["above_ma20"] = bars["close"] > bars["ma20"]
        bars["is_20d_high"] = grouped["close"].transform(lambda s: s >= s.rolling(20, min_periods=1).max())
        bars["is_60d_high"] = grouped["close"].transform(lambda s: s >= s.rolling(60, min_periods=1).max())

        snapshot = bars[bars["trade_date"] == signal_date].copy()
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])
        for window in [5, 10, 20, 60]:
            snapshot[f"rps_{window}"] = self._calculate_signal_date_rps(snapshot[f"return_{window}d"], signal_date, window)

        index_return = 0.0
        if index_bars is not None and not index_bars.empty and "close" in index_bars.columns:
            idx = index_bars.copy()
            idx["trade_date"] = pd.to_datetime(idx["trade_date"]).dt.normalize()
            if "index_code" in idx.columns:
                idx = idx.sort_values(["index_code", "trade_date"]).groupby("index_code", group_keys=False).head(len(idx))
                idx["index_return_3d"] = idx.groupby("index_code")["close"].pct_change(3)
            else:
                idx = idx.sort_values("trade_date")
                idx["index_return_3d"] = idx["close"].pct_change(3)
            current = idx[idx["trade_date"] == signal_date]
            if not current.empty:
                index_return = float(current["index_return_3d"].dropna().mean() or 0.0)
        snapshot["relative_return_vs_index"] = snapshot["return_3d"].fillna(0) - index_return

        snapshot["relative_return_vs_sector"] = 0.0
        if (
            sector_daily is not None
            and sector_map is not None
            and not sector_daily.empty
            and not sector_map.empty
            and "sector" in sector_daily.columns
            and "sector" in sector_map.columns
        ):
            sector = sector_daily.copy()
            sector["trade_date"] = pd.to_datetime(sector["trade_date"]).dt.normalize()
            if "sector_return_3d" not in sector.columns and "close" in sector.columns:
                sector = sector.sort_values(["sector", "trade_date"])
                sector["sector_return_3d"] = sector.groupby("sector")["close"].pct_change(3)
            current_sector = sector[sector["trade_date"] == signal_date][["sector", "sector_return_3d"]]
            snapshot = snapshot.merge(sector_map[["stock_code", "sector"]], on="stock_code", how="left")
            snapshot = snapshot.merge(current_sector, on="sector", how="left")
            snapshot["relative_return_vs_sector"] = snapshot["return_3d"].fillna(0) - snapshot["sector_return_3d"].fillna(0)

        snapshot["rps_composite"] = snapshot.apply(self._rps_composite, axis=1).round(2)
        snapshot["trend_efficiency_score"] = _rank_score(self._winsorize(snapshot["trend_efficiency"]))
        snapshot["rps_pattern"] = np.select(
            [
                (snapshot["rps_5"] >= 90)
                & (snapshot["rps_10"] >= 85)
                & (snapshot["rps_20"] >= 75)
                & (snapshot["rps_5"] >= snapshot["rps_10"]),
                (snapshot["rps_20"] >= 80)
                & (snapshot["rps_10"] >= 75)
                & (snapshot["rps_5"] >= 60)
                & (snapshot["above_ma20"].fillna(False)),
            ],
            ["acceleration", "trend_pullback"],
            default="neutral",
        )
        score = (
            snapshot["rps_20"].fillna(50) * 0.35
            + snapshot["rps_60"].fillna(50) * 0.25
            + snapshot["trend_efficiency_score"].fillna(50) * 0.25
            + snapshot["is_60d_high"].astype(float) * 100 * 0.15
        )
        snapshot["score"] = score.clip(0, 100).round(2)
        columns = [
            "stock_code",
            "trade_date",
            "score",
            "return_1d",
            "return_3d",
            "return_5d",
            "return_10d",
            "return_20d",
            "return_60d",
            "relative_return_vs_index",
            "relative_return_vs_sector",
            "rps_5",
            "rps_10",
            "rps_20",
            "rps_60",
            "rps_composite",
            "volatility_20d",
            "trend_efficiency",
            "trend_efficiency_score",
            "ma20",
            "above_ma20",
            "rps_pattern",
            "is_20d_high",
            "is_60d_high",
        ]
        return snapshot[[column for column in columns if column in snapshot.columns]].reset_index(drop=True)

    @staticmethod
    def _winsorize(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        if values.notna().sum() < 3:
            return values
        lower = values.quantile(0.05)
        upper = values.quantile(0.95)
        return values.clip(lower=lower, upper=upper)

    @staticmethod
    def _calculate_signal_date_rps(returns: pd.Series, signal_date: pd.Timestamp, window: int) -> pd.Series:
        valid = pd.to_numeric(returns, errors="coerce")
        valid_count = int(valid.notna().sum())
        if valid_count < MIN_CROSS_SECTION:
            logger.warning(
                "RPS fallback to neutral: trade_date=%s window=%s valid_count=%s min_cross_section=%s",
                pd.Timestamp(signal_date).date().isoformat(),
                window,
                valid_count,
                MIN_CROSS_SECTION,
            )
            return pd.Series(50.0, index=returns.index)
        return valid.rank(pct=True, na_option="keep") * 100

    @staticmethod
    def _rps_composite(row: pd.Series) -> float:
        weighted_sum = 0.0
        available_weight = 0.0
        for column, weight in RPS_WEIGHTS.items():
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                weighted_sum += float(value) * weight
                available_weight += weight
        if available_weight == 0:
            return 50.0
        return weighted_sum / available_weight
