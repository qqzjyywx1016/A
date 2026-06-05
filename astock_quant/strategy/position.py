"""Position sizing rules."""

from __future__ import annotations

from typing import Any

import pandas as pd


class PositionSizer:
    """Recommend total and single-stock position sizes by market regime and rating."""

    REGIME_TOTAL_POSITION = {
        "strong": 0.80,
        "neutral": 0.50,
        "weak": 0.20,
        "risk_off": 0.0,
    }

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def suggest(self, candidates: pd.DataFrame, *, market_regime: str = "neutral") -> pd.DataFrame:
        """Return candidates with suggested single-stock and target total position."""

        result = candidates.copy()
        total_position = self.REGIME_TOTAL_POSITION.get(market_regime, 0.40)
        max_single = float(self.config.get("max_single_position", 0.20))
        rating_position = {
            "A": float(self.config.get("rating_a_position", 0.15)),
            "B": float(self.config.get("rating_b_position", 0.10)),
            "C": float(self.config.get("rating_c_position", 0.05)),
        }
        result["target_total_position"] = total_position
        result["suggested_position"] = result["rating"].map(rating_position).fillna(0.0).clip(upper=max_single)
        total_single = result["suggested_position"].sum()
        if total_single > total_position and total_single > 0:
            result["suggested_position"] = result["suggested_position"] * (total_position / total_single)
        return result
