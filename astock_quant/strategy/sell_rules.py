"""Rule-based sell signal engine."""

from __future__ import annotations

from typing import Any

import pandas as pd


class SellRuleEngine:
    """Evaluate stop loss, take profit, trend and time-exit rules."""

    def __init__(self, config: dict[str, Any]):
        max_holding_days = config.get("max_holding_days", 3)
        self.max_holding_days = None if max_holding_days is None else int(max_holding_days)
        self.stop_loss = float(config.get("stop_loss", -0.05))
        self.take_profit = float(config.get("take_profit", 0.10))
        trail_pct = config.get("trail_pct")
        self.trail_pct = None if trail_pct is None else float(trail_pct)
        ma_exit_period = config.get("ma_exit_period")
        self.ma_exit_period = None if ma_exit_period is None else int(ma_exit_period)

    def evaluate(self, position: dict[str, Any], market_row: pd.Series, *, holding_days: int) -> str | None:
        """Return a sell reason or None if the position should continue holding."""

        entry_price = float(position["entry_price"])
        close = float(market_row["close"])
        trade_return = close / entry_price - 1
        if trade_return <= self.stop_loss:
            return "stop_loss"
        if self.trail_pct is not None:
            peak_close = float(position.get("peak_close", entry_price))
            if close <= peak_close * (1 - self.trail_pct):
                return "trailing_stop"
        if trade_return >= self.take_profit:
            return "take_profit"
        if self.ma_exit_period is not None:
            ma_column = f"ma{self.ma_exit_period}"
            if ma_column in market_row and pd.notna(market_row[ma_column]) and close < float(market_row[ma_column]):
                return "trend_break"
        sector_regime = str(market_row.get("active_sector_regime") or market_row.get("sector_regime", "")).lower()
        if sector_regime in {"weak", "risk_off"}:
            return "sector_fade"
        if self.max_holding_days is not None and holding_days >= self.max_holding_days:
            return "max_holding_days"
        return None
