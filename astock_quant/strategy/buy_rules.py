"""Rule-based buy suggestion engine."""

from __future__ import annotations

import pandas as pd


class BuyRuleEngine:
    """Generate next-day buy suggestions without placing orders."""

    def generate(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Attach a suggestion column using auction, breakout and pullback rules."""

        if candidates.empty:
            result = candidates.copy()
            result["suggestion"] = []
            return result

        result = candidates.copy()

        def decide(row: pd.Series) -> str:
            auction_return = row.get("auction_open_return")
            if bool(row.get("is_one_price_limit_up", False)):
                return "avoid"
            if pd.notna(auction_return):
                if auction_return > 0.07:
                    return "avoid"
                if 0 <= auction_return <= 0.05:
                    return "auction_confirm"
            if bool(row.get("break_yesterday_high", False)) and row.get("volume_ratio_20d", 0) >= 1.2:
                return "intraday_breakout"
            pullback = bool(row.get("pullback_to_ma5", False)) or bool(row.get("pullback_to_ma10", False))
            if pullback and bool(row.get("shrink_volume", True)):
                return "pullback_buy"
            return "watch"

        result["suggestion"] = result.apply(decide, axis=1)
        return result
