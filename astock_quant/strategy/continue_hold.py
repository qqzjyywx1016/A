"""Continue-hold scoring for open positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ContinueHoldDecision:
    """Continue-hold score, component scores and action."""

    score: int
    action: str
    reason: str | None
    components: dict[str, int]


class ContinueHoldScorer:
    """Score whether a position still deserves to be held."""

    REQUIRED_COLUMNS = {"close", "ma5", "ma10", "rps_20", "sector_rps_5", "sector_rps_10", "market_regime"}

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.exit_below = int(self.config.get("exit_below", 6))

    def can_evaluate(self, market_row: pd.Series) -> bool:
        """Return True when the row has the core columns required for scoring."""

        return self.REQUIRED_COLUMNS.issubset(set(market_row.index))

    def evaluate(self, market_row: pd.Series) -> ContinueHoldDecision:
        """Return the 0-10 continue-hold score and action."""

        components = {
            "trend_score": self._trend_score(market_row),
            "medium_momentum_score": self._medium_momentum_score(market_row),
            "sector_state_score": self._sector_state_score(market_row),
            "volume_price_health_score": self._volume_price_health_score(market_row),
            "risk_score": self._risk_score(market_row),
        }
        score = int(sum(components.values()))
        if score >= int(self.config.get("strong_hold_min", 8)):
            return ContinueHoldDecision(score, "strong_hold", None, components)
        if score >= self.exit_below:
            return ContinueHoldDecision(score, "hold_watch", None, components)
        return ContinueHoldDecision(score, "exit", "low_continue_hold_score", components)

    @staticmethod
    def _trend_score(row: pd.Series) -> int:
        close = float(row["close"])
        ma5 = float(row["ma5"])
        ma10 = float(row["ma10"])
        if close > ma5:
            return 2
        if ma5 >= close > ma10:
            return 1
        return 0

    @staticmethod
    def _medium_momentum_score(row: pd.Series) -> int:
        rps_20 = pd.to_numeric(row.get("rps_20"), errors="coerce")
        if pd.isna(rps_20):
            return 1
        if rps_20 >= 80:
            return 2
        if rps_20 >= 65:
            return 1
        return 0

    @staticmethod
    def _sector_state_score(row: pd.Series) -> int:
        sector_rps_5 = pd.to_numeric(row.get("sector_rps_5"), errors="coerce")
        sector_rps_10 = pd.to_numeric(row.get("sector_rps_10"), errors="coerce")
        if pd.isna(sector_rps_5):
            sector_rps_5 = 0
        if pd.isna(sector_rps_10):
            sector_rps_10 = 0
        if sector_rps_5 >= 70 and sector_rps_10 >= 65:
            return 2
        if sector_rps_5 >= 50 or sector_rps_10 >= 50:
            return 1
        return 0

    @staticmethod
    def _volume_price_health_score(row: pd.Series) -> int:
        high_volume_bearish = bool(row.get("high_volume_bearish", False))
        high_volume_stagnation = bool(row.get("high_volume_stagnation", False))
        long_upper_shadow = bool(row.get("long_upper_shadow", False))
        if high_volume_bearish or high_volume_stagnation:
            return 0
        if long_upper_shadow:
            return 0
        return 2

    @staticmethod
    def _risk_score(row: pd.Series) -> int:
        market_regime = str(row.get("market_regime", "neutral")).lower()
        major_risk = bool(row.get("is_major_event", False)) or bool(row.get("is_restructuring", False))
        if market_regime == "risk_off" or major_risk:
            return 0
        if market_regime == "weak":
            return 1
        return 2
