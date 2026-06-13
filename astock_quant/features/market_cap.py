"""Market-cap elasticity factor."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from astock_quant.features.momentum import _rank_score
from astock_quant.utils.calendar import ensure_no_future_data


class MarketCapFactor:
    """Score small and mid-cap tradable names with liquidity support.

    The thresholds are provisional, pending IC validation, and are intended to
    favor 30-80B RMB float-market-cap names after the universe floor has already
    removed smaller stocks.
    """

    def calculate(
        self,
        snapshot: pd.DataFrame,
        *,
        trade_date: str,
        sector_regime_map: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """Return market-cap elasticity scores for a T-day snapshot."""

        ensure_no_future_data(snapshot, trade_date)
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score", "market_cap_score"])

        result = snapshot.copy()
        if "trade_date" in result.columns:
            result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
            signal_date = pd.Timestamp(trade_date).normalize()
            result = result[result["trade_date"] == signal_date].copy()
        else:
            result["trade_date"] = pd.Timestamp(trade_date).normalize()
        if result.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score", "market_cap_score"])

        result["float_market_cap"] = pd.to_numeric(result.get("float_market_cap"), errors="coerce")
        if result["float_market_cap"].isna().all():
            result["tier_score"] = 50.0
            result["sector_bonus"] = 0.0
            result["market_cap_score"] = 50.0
            result["score"] = 50.0
            columns = [
                "stock_code",
                "trade_date",
                "score",
                "market_cap_score",
                "float_market_cap",
                "tier_score",
                "sector_bonus",
            ]
            return result[[column for column in columns if column in result.columns]].reset_index(drop=True)
        cap = result["float_market_cap"]
        # Piecewise-linear interpolation between tier knots instead of step
        # functions, so 7.9B vs 8.1B float cap no longer jumps 15 points.
        tier_knots_cap = [3_000_000_000, 8_000_000_000, 20_000_000_000, 50_000_000_000]
        tier_knots_score = [100.0, 85.0, 65.0, 50.0]
        result["tier_score"] = np.where(
            cap.notna(),
            np.interp(cap.fillna(0.0), tier_knots_cap, tier_knots_score),
            50.0,
        )

        amount_score = self._rank_or_neutral(result.get("avg_turnover_amount_20d"), result.index)
        turnover_rate_score = self._rank_or_neutral(result.get("avg_turnover_rate_20d"), result.index)
        result["sector_bonus"] = self._sector_bonus(result, sector_regime_map)
        result["market_cap_score"] = (
            result["tier_score"] * 0.60
            + amount_score * 0.20
            + turnover_rate_score * 0.10
            + result["sector_bonus"] * 0.10
        ).clip(0, 100).round(2)
        result["score"] = result["market_cap_score"]

        columns = [
            "stock_code",
            "trade_date",
            "score",
            "market_cap_score",
            "float_market_cap",
            "tier_score",
            "sector_bonus",
        ]
        return result[[column for column in columns if column in result.columns]].reset_index(drop=True)

    @staticmethod
    def _rank_or_neutral(values: object, index: pd.Index) -> pd.Series:
        if not isinstance(values, pd.Series):
            return pd.Series(50.0, index=index)
        return _rank_score(values)

    @staticmethod
    def _sector_bonus(result: pd.DataFrame, sector_regime_map: Mapping[str, str] | None) -> pd.Series:
        if "active_sector_regime" in result.columns:
            regimes = result["active_sector_regime"].astype(str).str.lower()
        elif sector_regime_map and "active_sector_code" in result.columns:
            regimes = result["active_sector_code"].map(sector_regime_map).fillna("").astype(str).str.lower()
        elif sector_regime_map and "sector_code" in result.columns:
            regimes = result["sector_code"].map(sector_regime_map).fillna("").astype(str).str.lower()
        else:
            regimes = pd.Series("", index=result.index)
        cap = result["float_market_cap"]
        return pd.Series(
            np.where((regimes == "strong") & (cap >= 3_000_000_000) & (cap < 8_000_000_000), 100.0, 0.0),
            index=result.index,
        )
