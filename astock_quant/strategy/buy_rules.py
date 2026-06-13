"""Rule-based buy planning engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BuyRuleEngine:
    """Generate executable next-day buy plans from T-day close data.

    A close-of-day system cannot observe tomorrow's auction or intraday prices,
    so instead of pretending to know them it emits concrete price levels the
    trader (or an execution layer) checks against tomorrow's tape: an auction
    confirmation band, a gap-up avoidance ceiling, the breakout trigger level
    and pullback support levels.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.auction_confirm_min_pct = float(self.config.get("auction_confirm_min_pct", 0.0))
        self.auction_confirm_max_pct = float(self.config.get("auction_confirm_max_pct", 0.05))
        self.avoid_gap_up_pct = float(self.config.get("avoid_gap_up_pct", 0.07))

    def generate(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Attach a plan type and next-day price levels to each candidate."""

        result = candidates.copy()
        plan_columns = [
            "suggestion",
            "buy_zone_low",
            "buy_zone_high",
            "avoid_above",
            "breakout_level",
            "pullback_support_ma5",
            "pullback_support_ma10",
        ]
        if result.empty:
            for column in plan_columns:
                result[column] = []
            return result

        close = pd.to_numeric(result.get("close"), errors="coerce") if "close" in result.columns else pd.Series(np.nan, index=result.index)
        result["buy_zone_low"] = (close * (1 + self.auction_confirm_min_pct)).round(2)
        result["buy_zone_high"] = (close * (1 + self.auction_confirm_max_pct)).round(2)
        result["avoid_above"] = (close * (1 + self.avoid_gap_up_pct)).round(2)
        result["breakout_level"] = (
            pd.to_numeric(result.get("high"), errors="coerce").round(2) if "high" in result.columns else np.nan
        )
        result["pullback_support_ma5"] = (
            pd.to_numeric(result.get("ma5"), errors="coerce").round(2) if "ma5" in result.columns else np.nan
        )
        result["pullback_support_ma10"] = (
            pd.to_numeric(result.get("ma10"), errors="coerce").round(2) if "ma10" in result.columns else np.nan
        )
        result["suggestion"] = result.apply(self._plan_type, axis=1)
        return result

    @staticmethod
    def _plan_type(row: pd.Series) -> str:
        """Choose the plan style from the T-day setup, all fields observable at close."""

        rps_pattern = str(row.get("rps_pattern", "") or "")
        if rps_pattern == "acceleration":
            return "breakout_buy_plan"
        if rps_pattern == "trend_pullback":
            return "pullback_buy_plan"
        reversal_score = pd.to_numeric(row.get("reversal_score"), errors="coerce")
        if pd.notna(reversal_score) and reversal_score >= 70:
            return "pullback_buy_plan"
        return "auction_confirm_plan"
