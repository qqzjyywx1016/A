"""Backtest trade record types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Trade:
    """Single backtest execution record."""

    trade_date: str
    stock_code: str
    side: str
    price: float
    shares: int
    amount: float
    fee: float
    tax: float
    reason: str
    pnl: float = 0.0
    holding_days: int = 0
