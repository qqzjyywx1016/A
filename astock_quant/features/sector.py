"""Sector resonance factor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from astock_quant.features.momentum import _rank_score
from astock_quant.utils.calendar import ensure_no_future_data
from astock_quant.utils.logger import get_logger


logger = get_logger(__name__)
DEFAULT_SECTOR_RPS_WEIGHTS = {
    "sector_rps_3": 0.35,
    "sector_rps_5": 0.30,
    "sector_rps_10": 0.25,
    "sector_rps_20": 0.10,
}


class SectorFactor:
    """Score stocks by sector strength and in-sector leadership."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.min_cross_section = int(self.config.get("min_cross_section", 5))
        self.composite_weights = dict(self.config.get("composite_weights", DEFAULT_SECTOR_RPS_WEIGHTS))

    def calculate(
        self,
        daily_bars: pd.DataFrame,
        *,
        trade_date: str,
        sector_map: pd.DataFrame | None = None,
        sector_daily: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return one sector score row per stock for the signal date."""

        ensure_no_future_data(daily_bars, trade_date)
        if sector_daily is not None:
            ensure_no_future_data(sector_daily, trade_date)
        if daily_bars.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        bars = daily_bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        signal_date = pd.Timestamp(trade_date).normalize()

        bars = bars.sort_values(["stock_code", "trade_date"])
        bars["stock_return_3d"] = bars.groupby("stock_code")["close"].pct_change(3)
        snapshot = bars[bars["trade_date"] == signal_date].copy()
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        membership = self._build_stock_sector_membership(snapshot, sector_map)
        if sector_daily is None or sector_daily.empty:
            logger.warning("sector_daily missing; using neutral sector RPS and score")
            return self._neutral_result(membership)

        current_sector = self._current_sector_snapshot(sector_daily, signal_date)
        if current_sector.empty:
            logger.warning("sector_daily missing signal date %s; using neutral sector RPS and score", trade_date)
            return self._neutral_result(membership)

        total_amount = current_sector["sector_turnover_amount"].sum()
        current_sector["sector_amount_ratio"] = np.where(
            total_amount > 0, current_sector["sector_turnover_amount"] / total_amount, 0.0
        )
        current_sector["sector_rank_pct"] = current_sector["sector_rps_composite"].fillna(50) / 100
        current_sector["sector_regime"] = np.select(
            [
                current_sector["sector_rps_composite"].fillna(50) >= 70,
                current_sector["sector_rps_composite"].fillna(50) < 50,
            ],
            ["strong", "weak"],
            default="neutral",
        )

        snapshot = membership.merge(current_sector.drop(columns=["sector_name", "sector_type"]), on="sector_code", how="left")
        snapshot["stock_rank_in_sector"] = snapshot.groupby("sector_code")["stock_return_3d"].rank(pct=True).fillna(0.5)
        strong_counts = (
            snapshot.assign(is_strong=snapshot["stock_return_3d"].fillna(0) > 0.03)
            .groupby("sector_code")["is_strong"]
            .sum()
            .rename("sector_strong_stock_count")
        )
        snapshot = snapshot.merge(strong_counts, on="sector_code", how="left")
        count_score = (snapshot["sector_strong_stock_count"].fillna(0).clip(0, 10) / 10) * 100
        sector_amount_score = _rank_score(snapshot["sector_amount_ratio"].fillna(0))
        stock_rank_score = snapshot["stock_rank_in_sector"].fillna(0.5) * 100
        snapshot["score"] = (
            snapshot["sector_rps_composite"].fillna(50) * 0.50
            + sector_amount_score * 0.20
            + stock_rank_score * 0.20
            + count_score * 0.10
        ).clip(0, 100).round(2)
        snapshot = self._select_active_sector(snapshot)
        columns = [
            "stock_code",
            "trade_date",
            "score",
            "sector",
            "sector_code",
            "sector_name",
            "sector_type",
            "active_sector_code",
            "active_sector_name",
            "active_sector_type",
            "active_sector_rps",
            "sector_return_1d",
            "sector_return_3d",
            "sector_return_5d",
            "sector_return_10d",
            "sector_return_20d",
            "sector_rps_3",
            "sector_rps_5",
            "sector_rps_10",
            "sector_rps_20",
            "sector_rps_composite",
            "sector_rps_pattern",
            "sector_amount_ratio",
            "sector_rank_pct",
            "stock_rank_in_sector",
            "sector_strong_stock_count",
            "sector_regime",
        ]
        return snapshot[columns].reset_index(drop=True)

    def _current_sector_snapshot(self, sector_daily: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
        sector = self._normalize_sector_frame(sector_daily)
        sector["trade_date"] = pd.to_datetime(sector["trade_date"]).dt.normalize()
        sector = sector.sort_values(["sector_code", "trade_date"])
        if "turnover_amount" not in sector.columns:
            sector["turnover_amount"] = 0.0
        if "sector_return_1d" not in sector.columns:
            sector["sector_return_1d"] = (
                sector.groupby("sector_code")["close"].pct_change(1) if "close" in sector.columns else 0.0
            )
        for window in [3, 5, 10, 20]:
            return_column = f"sector_return_{window}d"
            if return_column not in sector.columns:
                sector[return_column] = (
                    sector.groupby("sector_code")["close"].pct_change(window) if "close" in sector.columns else np.nan
                )
        current = sector[sector["trade_date"] == signal_date].copy()
        if current.empty:
            return current
        current = current.rename(columns={"turnover_amount": "sector_turnover_amount"})
        for window in [3, 5, 10, 20]:
            current[f"sector_rps_{window}"] = self._signal_date_rps(current[f"sector_return_{window}d"], window)
        current["sector_rps_composite"] = current.apply(self._sector_rps_composite, axis=1).round(2)
        current["sector_rps_pattern"] = np.select(
            [
                (current["sector_rps_3"] >= 85)
                & (current["sector_rps_5"] >= 80)
                & (current["sector_rps_10"] >= 75)
                & (current["sector_rps_3"] >= current["sector_rps_5"]),
                (current["sector_rps_10"] >= 80) & (current["sector_rps_20"] >= 75) & (current["sector_rps_5"] >= 65),
                (current["sector_rps_5"] < 50) & (current["sector_rps_10"] < 50),
            ],
            ["sector_acceleration", "sector_trend", "sector_weak"],
            default="neutral",
        )
        return current[
            [
                "sector_code",
                "sector_name",
                "sector_type",
                "sector_return_1d",
                "sector_return_3d",
                "sector_return_5d",
                "sector_return_10d",
                "sector_return_20d",
                "sector_rps_3",
                "sector_rps_5",
                "sector_rps_10",
                "sector_rps_20",
                "sector_rps_composite",
                "sector_rps_pattern",
                "sector_turnover_amount",
            ]
        ]

    def _signal_date_rps(self, returns: pd.Series, window: int) -> pd.Series:
        values = pd.to_numeric(returns, errors="coerce")
        valid_count = int(values.notna().sum())
        if valid_count < self.min_cross_section:
            logger.warning(
                "Sector RPS fallback to neutral: window=%s valid_count=%s min_cross_section=%s",
                window,
                valid_count,
                self.min_cross_section,
            )
            return pd.Series(50.0, index=returns.index)
        return values.rank(pct=True, na_option="keep") * 100

    def _sector_rps_composite(self, row: pd.Series) -> float:
        weighted_sum = 0.0
        available_weight = 0.0
        for column, weight in self.composite_weights.items():
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                weighted_sum += float(value) * float(weight)
                available_weight += float(weight)
        if available_weight == 0:
            return 50.0
        return weighted_sum / available_weight

    def _build_stock_sector_membership(self, snapshot: pd.DataFrame, sector_map: pd.DataFrame | None) -> pd.DataFrame:
        base = snapshot.copy()
        if sector_map is not None and not sector_map.empty and "stock_code" in sector_map.columns:
            mapping = self._normalize_sector_frame(sector_map)
            membership = base.merge(
                mapping[["stock_code", "sector_code", "sector_name", "sector_type"]].drop_duplicates(),
                on="stock_code",
                how="left",
            )
        else:
            membership = base.copy()
            if "sector" not in membership.columns:
                membership["sector"] = "UNKNOWN"
            membership = self._normalize_sector_frame(membership)
        membership["sector_code"] = membership["sector_code"].fillna("UNKNOWN")
        membership["sector_name"] = membership["sector_name"].fillna(membership["sector_code"])
        membership["sector_type"] = membership["sector_type"].fillna("unknown")
        membership["sector"] = membership["sector_name"]
        return membership

    @staticmethod
    def _normalize_sector_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "sector_code" not in result.columns:
            if "sector" in result.columns:
                result["sector_code"] = result["sector"]
            elif "sector_name" in result.columns:
                result["sector_code"] = result["sector_name"]
            else:
                result["sector_code"] = "UNKNOWN"
        if "sector_name" not in result.columns:
            if "sector" in result.columns:
                result["sector_name"] = result["sector"]
            else:
                result["sector_name"] = result["sector_code"]
        if "sector_type" not in result.columns:
            result["sector_type"] = "unknown"
        if "sector" not in result.columns:
            result["sector"] = result["sector_name"]
        return result

    def _neutral_result(self, membership: pd.DataFrame) -> pd.DataFrame:
        result = self._select_active_sector(membership.copy())
        result["score"] = 50.0
        for column in ["sector_return_3d", "sector_return_5d", "sector_return_10d", "sector_return_20d"]:
            result[column] = np.nan
        result["sector_return_1d"] = np.nan
        for column in ["sector_rps_3", "sector_rps_5", "sector_rps_10", "sector_rps_20", "sector_rps_composite"]:
            result[column] = 50.0
        result["active_sector_rps"] = 50.0
        result["sector_rps_pattern"] = "unknown"
        result["sector_amount_ratio"] = 0.0
        result["sector_rank_pct"] = 0.5
        result["stock_rank_in_sector"] = 0.5
        result["sector_strong_stock_count"] = 0
        result["sector_regime"] = "neutral"
        columns = [
            "stock_code",
            "trade_date",
            "score",
            "sector",
            "sector_code",
            "sector_name",
            "sector_type",
            "active_sector_code",
            "active_sector_name",
            "active_sector_type",
            "active_sector_rps",
            "sector_return_1d",
            "sector_return_3d",
            "sector_return_5d",
            "sector_return_10d",
            "sector_return_20d",
            "sector_rps_3",
            "sector_rps_5",
            "sector_rps_10",
            "sector_rps_20",
            "sector_rps_composite",
            "sector_rps_pattern",
            "sector_amount_ratio",
            "sector_rank_pct",
            "stock_rank_in_sector",
            "sector_strong_stock_count",
            "sector_regime",
        ]
        return result[columns].reset_index(drop=True)

    @staticmethod
    def _select_active_sector(snapshot: pd.DataFrame) -> pd.DataFrame:
        result = snapshot.copy()
        if "sector_rps_composite" not in result.columns:
            result["sector_rps_composite"] = 50.0
        if "score" not in result.columns:
            result["score"] = 50.0
        result = result.sort_values(["stock_code", "sector_rps_composite", "score"], ascending=[True, False, False])
        result = result.groupby("stock_code", as_index=False, group_keys=False).head(1).copy()
        result["active_sector_code"] = result["sector_code"]
        result["active_sector_name"] = result["sector_name"]
        result["active_sector_type"] = result["sector_type"]
        result["active_sector_rps"] = result["sector_rps_composite"]
        result["sector"] = result["sector_name"]
        return result
