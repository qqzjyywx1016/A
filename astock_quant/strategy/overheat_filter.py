"""Pre-selection overheat filter for short-term entry risk."""

from __future__ import annotations

from typing import Any

import pandas as pd


class OverheatFilter:
    """Remove overheated candidates before final ranking and trade planning."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return passed candidates and rejected candidates with reject_reason."""

        if frame.empty:
            return frame.copy(), pd.DataFrame(columns=["stock_code", "reject_reason"])

        data = frame.copy()
        data["_pct_chg"] = pd.to_numeric(self._series(data, "return_1d", 0), errors="coerce").fillna(0) * 100
        amount_source = "turnover_amount" if "turnover_amount" in data.columns else "amount"
        data["_amount"] = pd.to_numeric(self._series(data, amount_source, pd.NA), errors="coerce")
        data["_amount_60d_max"] = pd.to_numeric(self._series(data, "amount_60d_max", pd.NA), errors="coerce")
        sector_key = self._sector_key(data)
        data["_sector_stock_count"] = data.groupby(sector_key, dropna=False)["stock_code"].transform("count")
        if "is_limit_up" in data.columns:
            data["_is_limit_up"] = data["is_limit_up"].fillna(False).astype(bool)
            data["_sector_limit_up_ratio"] = data.groupby(sector_key, dropna=False)["_is_limit_up"].transform("mean")
        else:
            data["_sector_limit_up_ratio"] = 0.0

        reasons = []
        for _, row in data.iterrows():
            reasons.append(self._reject_reason(row))
        data["reject_reason"] = reasons
        rejected = data[data["reject_reason"].notna()].copy()
        passed = data[data["reject_reason"].isna()].copy()
        helper_columns = [
            "_pct_chg",
            "_amount",
            "_amount_60d_max",
            "_sector_stock_count",
            "_is_limit_up",
            "_sector_limit_up_ratio",
        ]
        passed = passed.drop(columns=[column for column in helper_columns + ["reject_reason"] if column in passed.columns])
        rejected = rejected.drop(columns=[column for column in helper_columns if column in rejected.columns])
        if rejected.empty:
            rejected = pd.DataFrame(columns=list(frame.columns) + ["reject_reason"])
        return passed.reset_index(drop=True), rejected.reset_index(drop=True)

    @staticmethod
    def _sector_key(data: pd.DataFrame) -> pd.Series:
        if "active_sector_code" in data.columns:
            return data["active_sector_code"].fillna("UNKNOWN")
        if "sector" in data.columns:
            return data["sector"].fillna("UNKNOWN")
        return pd.Series("ALL", index=data.index)

    def _reject_reason(self, row: pd.Series) -> str | None:
        rps_5 = self._num(row.get("rps_5"))
        rps_10 = self._num(row.get("rps_10"))
        pct_chg = self._num(row.get("_pct_chg"))
        volume_ratio = self._num(row.get("volume_ratio_20d"))
        sector_rps_5 = self._num(row.get("sector_rps_5"))
        sector_stock_count = self._num(row.get("_sector_stock_count"))
        sector_limit_up_ratio = self._num(row.get("_sector_limit_up_ratio"))

        if (
            rps_5 >= self._cfg("rps5_threshold", 95)
            and pct_chg >= self._cfg("large_gain_pct", 7)
            and volume_ratio >= self._cfg("volume_ratio_threshold", 2.5)
        ):
            return "overheat_rps5_large_gain_volume"
        if (
            rps_5 >= self._cfg("climax_rps5_threshold", 97)
            and rps_10 >= self._cfg("climax_rps10_threshold", 95)
            and pct_chg >= self._cfg("climax_gain_pct", 8)
            and pd.notna(row.get("_amount_60d_max"))
            and self._num(row.get("_amount")) >= self._num(row.get("_amount_60d_max"))
        ):
            return "climax_acceleration"
        if (
            volume_ratio >= self._cfg("stagnation_volume_ratio", 3)
            and pct_chg < self._cfg("stagnation_gain_pct", 3)
            and row.get("long_upper_shadow") is True
        ):
            return "high_volume_stagnation"
        if (
            sector_stock_count >= self._cfg("sector_stock_count_threshold", 10)
            and sector_rps_5 >= self._cfg("sector_rps5_threshold", 95)
            and sector_limit_up_ratio >= self._cfg("sector_limit_up_ratio_threshold", 0.08)
        ):
            return "sector_climax"
        return None

    def _cfg(self, key: str, default: float) -> float:
        return float(self.config.get(key, default))

    @staticmethod
    def _series(data: pd.DataFrame, column: str, default: object) -> pd.Series:
        if column in data.columns:
            return data[column]
        return pd.Series(default, index=data.index)

    @staticmethod
    def _num(value: object) -> float:
        parsed = pd.to_numeric(value, errors="coerce")
        if pd.isna(parsed):
            return float("nan")
        return float(parsed)
