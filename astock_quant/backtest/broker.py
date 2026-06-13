"""Simple A-share execution cost model."""

from __future__ import annotations


class Broker:
    """Apply commission, stamp tax and slippage to buy and sell executions."""

    def __init__(
        self,
        commission_rate: float,
        stamp_tax_rate: float,
        slippage_rate: float,
        min_commission: float = 0.0,
    ):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.min_commission = float(min_commission)

    def buy_price(self, open_price: float) -> float:
        """Return buy price after positive slippage."""

        return float(open_price) * (1 + self.slippage_rate)

    def sell_price(self, close_price: float) -> float:
        """Return sell price after negative slippage."""

        return float(close_price) * (1 - self.slippage_rate)

    def commission(self, amount: float) -> float:
        """Return commission for a notional amount, honoring the broker minimum."""

        raw = abs(float(amount)) * self.commission_rate
        if raw <= 0:
            return raw
        return max(raw, self.min_commission)

    def stamp_tax(self, amount: float) -> float:
        """Return sell-side stamp tax for a notional amount."""

        return abs(float(amount)) * self.stamp_tax_rate
