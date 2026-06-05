"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MetricsCalculator:
    """Calculate return, risk and trade-quality metrics from backtest outputs."""

    def calculate(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        *,
        benchmark_curve: pd.DataFrame | None = None,
    ) -> dict[str, float | int]:
        """Return performance metrics required by the strategy spec."""

        if equity_curve.empty:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_holding_days": 0.0,
                "avg_trade_return": 0.0,
                "max_single_trade_loss": 0.0,
                "consecutive_loss_count": 0,
                "turnover_rate": 0.0,
                "benchmark_excess_return": 0.0,
            }

        curve = equity_curve.copy()
        curve["trade_date"] = pd.to_datetime(curve["trade_date"])
        start_equity = float(curve["equity"].iloc[0])
        end_equity = float(curve["equity"].iloc[-1])
        total_return = end_equity / start_equity - 1 if start_equity else 0.0
        days = max((curve["trade_date"].iloc[-1] - curve["trade_date"].iloc[0]).days, 1)
        annual_return = (1 + total_return) ** (365 / days) - 1
        peak = curve["equity"].cummax()
        drawdown = curve["equity"] / peak - 1
        max_drawdown = float(drawdown.min())

        sells = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
        if not sells.empty:
            pnl = pd.to_numeric(sells["pnl"], errors="coerce").fillna(0.0)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            win_rate = float((pnl > 0).mean())
            profit_loss_ratio = float(wins.mean() / abs(losses.mean())) if not losses.empty and losses.mean() != 0 else float("inf")
            avg_holding_days = float(pd.to_numeric(sells["holding_days"], errors="coerce").fillna(0).mean())
            sell_amount = pd.to_numeric(sells["amount"], errors="coerce").abs().replace(0, np.nan)
            avg_trade_return = float((pnl / sell_amount).replace([np.inf, -np.inf], np.nan).fillna(0).mean())
            max_single_trade_loss = float(pnl.min())
            consecutive_loss_count = self._max_consecutive_losses(pnl)
        else:
            win_rate = profit_loss_ratio = avg_holding_days = avg_trade_return = max_single_trade_loss = 0.0
            consecutive_loss_count = 0

        turnover = 0.0
        if not trades.empty and start_equity:
            turnover = float(pd.to_numeric(trades["amount"], errors="coerce").abs().sum() / start_equity)

        benchmark_excess_return = total_return
        if benchmark_curve is not None and not benchmark_curve.empty and "equity" in benchmark_curve.columns:
            benchmark_return = float(benchmark_curve["equity"].iloc[-1] / benchmark_curve["equity"].iloc[0] - 1)
            benchmark_excess_return = total_return - benchmark_return

        return {
            "total_return": round(float(total_return), 6),
            "annual_return": round(float(annual_return), 6),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(win_rate, 6),
            "profit_loss_ratio": round(profit_loss_ratio, 6) if np.isfinite(profit_loss_ratio) else float("inf"),
            "avg_holding_days": round(avg_holding_days, 3),
            "avg_trade_return": round(avg_trade_return, 6),
            "max_single_trade_loss": round(max_single_trade_loss, 2),
            "consecutive_loss_count": int(consecutive_loss_count),
            "turnover_rate": round(turnover, 6),
            "benchmark_excess_return": round(float(benchmark_excess_return), 6),
        }

    @staticmethod
    def _max_consecutive_losses(pnl: pd.Series) -> int:
        max_count = 0
        current = 0
        for value in pnl:
            if value < 0:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count
