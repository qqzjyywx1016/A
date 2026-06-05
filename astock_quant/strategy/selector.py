"""Stock selection rules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from astock_quant.utils.calendar import ensure_no_future_data
from astock_quant.utils.errors import DataQualityError
from astock_quant.utils.logger import get_logger


logger = get_logger(__name__)


class StockSelector:
    """Build core and watch pools from scored stocks."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rps_config = config.get("rps", {})
        self.sector_rps_config = config.get("sector_rps")

    def select(self, scored: pd.DataFrame, *, trade_date: str) -> dict[str, pd.DataFrame]:
        """Return core_pool and watch_pool DataFrames for the signal date."""

        ensure_no_future_data(scored, trade_date)
        if scored.empty:
            empty = scored.copy()
            return {"core_pool": empty, "watch_pool": empty}

        data = scored.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
        signal_date = pd.Timestamp(trade_date).normalize()
        data = data[data["trade_date"] == signal_date].copy()
        if data.empty:
            return {"core_pool": data, "watch_pool": data}

        if "sector_regime" not in data.columns:
            data["sector_regime"] = "neutral"
        if "long_upper_shadow" not in data.columns:
            data["long_upper_shadow"] = False
        if "high_volume_stagnation" not in data.columns:
            data["high_volume_stagnation"] = False
        if "market_regime" not in data.columns:
            logger.warning("market_regime missing from scored rows; defaulting to neutral")
            data["market_regime"] = "neutral"

        if self.rps_config.get("enabled", True):
            required_rps_columns = {
                "rps_20",
                "rps_60",
                "return_10d",
                "above_ma20",
            }
            missing = sorted(required_rps_columns.difference(data.columns))
            if missing:
                raise DataQualityError(f"missing required RPS columns: {', '.join(missing)}")

            if data["market_regime"].fillna("neutral").eq("risk_off").all():
                return {"core_pool": data.iloc[0:0].copy(), "watch_pool": data.iloc[0:0].copy()}
        if self._sector_rps_enabled():
            required_sector_rps_columns = {"sector_rps_5", "sector_rps_10"}
            missing = sorted(required_sector_rps_columns.difference(data.columns))
            if missing:
                raise DataQualityError(f"missing required sector RPS columns: {', '.join(missing)}")
            if data["market_regime"].fillna("neutral").eq("risk_off").all():
                return {"core_pool": data.iloc[0:0].copy(), "watch_pool": data.iloc[0:0].copy()}

        min_total_score = self.config.get("min_total_score", 70)
        candidates = data[
            (data["total_score"] >= min_total_score)
            & (data["rating"].isin(["A", "B"]))
            & (data["sector_regime"].fillna("neutral").isin(["strong", "neutral"]))
            & (~data["long_upper_shadow"].fillna(False).astype(bool))
            & (~data["high_volume_stagnation"].fillna(False).astype(bool))
        ].copy()
        if self.rps_config.get("enabled", True) and not candidates.empty:
            candidates = self._apply_rps_filters(candidates)
        if self._sector_rps_enabled() and not candidates.empty:
            candidates = self._apply_sector_rps_filters(candidates)
        candidates = candidates.sort_values("total_score", ascending=False).reset_index(drop=True)
        candidates = self._apply_max_per_sector(candidates)
        watch_pool = candidates.head(self.config.get("max_candidates", 20)).reset_index(drop=True)
        core_pool = watch_pool[watch_pool["rating"] == "A"].head(self.config.get("max_core_pool", 5)).reset_index(drop=True)
        return {"core_pool": core_pool, "watch_pool": watch_pool}

    def _apply_rps_filters(self, candidates: pd.DataFrame) -> pd.DataFrame:
        filters = self.rps_config.get("filters", {})
        gate = self.rps_config.get("gate", {})
        rps_60_min = float(gate.get("rps_60_min", 60))
        return_10d_min = float(gate.get("return_10d_min", -0.03))
        mask = pd.Series(True, index=candidates.index)
        for index, row in candidates.iterrows():
            regime = str(row.get("market_regime") or "neutral")
            if regime == "risk_off":
                mask.loc[index] = False
                continue
            thresholds = filters.get(regime, filters.get("neutral", {}))
            rps_20_threshold = thresholds.get("rps_20")
            rps_20 = pd.to_numeric(row.get("rps_20"), errors="coerce")
            rps_60 = pd.to_numeric(row.get("rps_60"), errors="coerce")
            return_10d = pd.to_numeric(row.get("return_10d", 0), errors="coerce")
            if rps_20_threshold is not None and (pd.isna(rps_20) or rps_20 < float(rps_20_threshold)):
                mask.loc[index] = False
            if pd.isna(rps_60) or rps_60 < rps_60_min:
                mask.loc[index] = False
            if pd.isna(return_10d) or return_10d <= return_10d_min:
                mask.loc[index] = False
            if row.get("above_ma20") is not True:
                mask.loc[index] = False
        return candidates.loc[mask].copy()

    def _sector_rps_enabled(self) -> bool:
        return isinstance(self.sector_rps_config, dict) and self.sector_rps_config.get("enabled", False)

    def _apply_sector_rps_filters(self, candidates: pd.DataFrame) -> pd.DataFrame:
        filters = self.sector_rps_config.get("filters", {}) if isinstance(self.sector_rps_config, dict) else {}
        mask = pd.Series(True, index=candidates.index)
        for index, row in candidates.iterrows():
            regime = str(row.get("market_regime") or "neutral")
            if regime == "risk_off":
                mask.loc[index] = False
                continue
            thresholds = filters.get(regime, filters.get("neutral", {}))
            for column in ["sector_rps_5", "sector_rps_10"]:
                threshold = thresholds.get(column)
                value = pd.to_numeric(row.get(column), errors="coerce")
                if threshold is not None and (pd.isna(value) or value < threshold):
                    mask.loc[index] = False
        return candidates.loc[mask].copy()

    def _apply_max_per_sector(self, candidates: pd.DataFrame) -> pd.DataFrame:
        max_per_sector = self.config.get("max_per_sector")
        if not max_per_sector or candidates.empty:
            return candidates
        key = None
        for column in ["active_sector_code", "active_sector_name", "sector"]:
            if column in candidates.columns:
                key = column
                break
        if key is None:
            return candidates
        limited = candidates.groupby(key, group_keys=False).head(int(max_per_sector)).reset_index(drop=True)
        return self._apply_max_sector_exposure(limited, key)

    def _apply_max_sector_exposure(self, candidates: pd.DataFrame, key: str) -> pd.DataFrame:
        max_sector_exposure = self.config.get("max_sector_exposure")
        if max_sector_exposure is None or "suggested_position" not in candidates.columns or candidates.empty:
            return candidates
        kept = []
        used: dict[Any, float] = {}
        for _, row in candidates.iterrows():
            sector_key = row.get(key)
            suggested = pd.to_numeric(row.get("suggested_position"), errors="coerce")
            suggested_value = 0.0 if pd.isna(suggested) else float(suggested)
            current = used.get(sector_key, 0.0)
            if current + suggested_value > float(max_sector_exposure):
                continue
            kept.append(row)
            used[sector_key] = current + suggested_value
        if not kept:
            return candidates.iloc[0:0].copy()
        return pd.DataFrame(kept).reset_index(drop=True)
