"""Composite factor scoring engine."""

from __future__ import annotations

import pandas as pd

from astock_quant.utils.logger import get_logger


logger = get_logger(__name__)


class ScoreEngine:
    """Combine factor scores with configurable weights and assign ratings."""

    FACTOR_COLUMNS = {
        "momentum": "momentum_score",
        "volume": "volume_score",
        "sector": "sector_score",
        "market_cap": "market_cap_score",
        "reversal": "reversal_score",
        "fund_flow": "fund_score",
        "pattern": "pattern_score",
        "sentiment": "sentiment_score",
    }

    def __init__(self, weights: dict[str, float], regime_multipliers: dict[str, dict[str, float]] | None = None):
        self.weights = weights
        self.regime_multipliers = regime_multipliers or {}

    @staticmethod
    def _rating(total_score: float) -> str:
        if total_score >= 80:
            return "A"
        if total_score >= 70:
            return "B"
        if total_score >= 60:
            return "C"
        return "D"

    def _base_frame(self, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for df in factors.values():
            if df is not None and not df.empty and {"stock_code", "trade_date"}.issubset(df.columns):
                frame = df[["stock_code", "trade_date"]].copy()
                frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["stock_code", "trade_date"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(["stock_code", "trade_date"])

    def score(
        self,
        factors: dict[str, pd.DataFrame],
        stock_basic: pd.DataFrame | None = None,
        market_regime: str | None = None,
    ) -> pd.DataFrame:
        """Return composite scores with required output fields and useful selection flags.

        ``market_regime`` (the day's history-only sentiment state) selects a row of
        ``regime_multipliers`` that scales the per-factor weights, so e.g. reversal
        can be throttled in regimes where its IC is not proven. Absent regime or
        multipliers, weighting is unchanged.
        """

        result = self._base_frame(factors)
        if result.empty:
            return pd.DataFrame(
                columns=[
                    "stock_code",
                    "stock_name",
                    "trade_date",
                    "sector",
                    "total_score",
                    "momentum_score",
                    "volume_score",
                    "sector_score",
                    "market_cap_score",
                    "reversal_score",
                    "fund_score",
                    "pattern_score",
                    "sentiment_score",
                    "rating",
                ]
            )

        passthrough_columns = {
            "momentum": [
                "rps_5",
                "rps_10",
                "rps_20",
                "rps_60",
                "rps_composite",
                "rps_pattern",
                "volatility_20d",
                "trend_efficiency",
                "trend_efficiency_score",
                "return_1d",
                "return_5d",
                "return_10d",
                "above_ma20",
                "ma20",
                "is_20d_high",
                "is_60d_high",
            ],
            "sector": [
                "sector",
                "sector_regime",
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
                "active_sector_code",
                "active_sector_name",
                "active_sector_type",
                "active_sector_rps",
            ],
            "volume": ["volume_ratio_20d", "turnover_amount", "amount_60d_max"],
            "market_cap": ["float_market_cap"],
            "reversal": ["pullback_score", "decel_score", "shrink_score"],
            "pattern": ["long_upper_shadow", "high_volume_stagnation", "high_volume_bearish"],
            "sentiment": ["market_regime"],
        }
        numeric_passthrough = {
            "rps_5",
            "rps_10",
            "rps_20",
            "rps_60",
            "rps_composite",
            "volatility_20d",
            "trend_efficiency",
            "trend_efficiency_score",
            "return_1d",
            "return_5d",
            "return_10d",
            "ma20",
            "volume_ratio_20d",
            "turnover_amount",
            "amount_60d_max",
            "sector_return_3d",
            "sector_return_5d",
            "sector_return_10d",
            "sector_return_20d",
            "sector_rps_3",
            "sector_rps_5",
            "sector_rps_10",
            "sector_rps_20",
            "sector_rps_composite",
            "active_sector_rps",
            "float_market_cap",
            "pullback_score",
            "decel_score",
            "shrink_score",
        }
        bool_passthrough = {
            "above_ma20",
            "is_20d_high",
            "is_60d_high",
            "long_upper_shadow",
            "high_volume_stagnation",
            "high_volume_bearish",
        }
        inactive_factors: set[str] = set()
        for factor_name, score_column in self.FACTOR_COLUMNS.items():
            df = factors.get(factor_name)
            if df is None or df.empty:
                result[score_column] = 50.0
                inactive_factors.add(factor_name)
                logger.warning("factor %s missing or empty; using neutral score 50", factor_name)
                continue
            keep = ["stock_code", "trade_date", "score"]
            keep.extend(column for column in passthrough_columns.get(factor_name, []) if column in df.columns)
            factor_frame = df[keep].copy().rename(columns={"score": score_column})
            factor_frame["trade_date"] = pd.to_datetime(factor_frame["trade_date"]).dt.normalize()
            result = result.merge(factor_frame, on=["stock_code", "trade_date"], how="left")
            result[score_column] = pd.to_numeric(result[score_column], errors="coerce").fillna(50.0)
            if result[score_column].eq(50.0).all():
                inactive_factors.add(factor_name)
                logger.warning("factor %s is all neutral 50; it may not be contributing", factor_name)
            for column in numeric_passthrough.intersection(result.columns):
                result[column] = pd.to_numeric(result[column], errors="coerce")
            for column in bool_passthrough.intersection(result.columns):
                result[column] = result[column].map(self._normalize_bool_passthrough).astype(object)

        if stock_basic is not None and not stock_basic.empty and "stock_code" in stock_basic.columns:
            metadata = stock_basic.copy()
            keep_meta = [column for column in ["stock_code", "stock_name", "sector"] if column in metadata.columns]
            result = result.merge(metadata[keep_meta].drop_duplicates("stock_code"), on="stock_code", how="left", suffixes=("", "_basic"))
            if "sector_basic" in result.columns:
                result["sector"] = result.get("sector").fillna(result["sector_basic"])
                result = result.drop(columns=["sector_basic"])

        for column in ["stock_name", "sector"]:
            if column not in result.columns:
                result[column] = ""
            result[column] = result[column].fillna("")

        regime_mult = self.regime_multipliers.get(str(market_regime), {}) if market_regime is not None else {}
        weighted_score = pd.Series(0.0, index=result.index)
        active_weight_sum = 0.0
        for factor_name, score_column in self.FACTOR_COLUMNS.items():
            weight = float(self.weights.get(factor_name, 0)) * float(regime_mult.get(factor_name, 1.0))
            if weight <= 0 or factor_name in inactive_factors:
                continue
            weighted_score += result[score_column] * weight
            active_weight_sum += weight
        if active_weight_sum > 0:
            result["total_score"] = (weighted_score / active_weight_sum).round(2)
        else:
            result["total_score"] = 50.0
        result["rating"] = result["total_score"].apply(self._rating)

        required_order = [
            "stock_code",
            "stock_name",
            "trade_date",
            "sector",
            "total_score",
            "momentum_score",
            "volume_score",
            "sector_score",
            "market_cap_score",
            "reversal_score",
            "fund_score",
            "pattern_score",
            "sentiment_score",
            "rating",
        ]
        extra = [column for column in result.columns if column not in required_order]
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.date.astype(str)
        return result[required_order + extra].sort_values("total_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _normalize_bool_passthrough(value: object) -> object:
        if pd.isna(value):
            return False
        if value is True:
            return True
        if value is False:
            return False
        return value
