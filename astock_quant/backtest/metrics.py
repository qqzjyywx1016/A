"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


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
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "drawdown_recovery_days": float("nan"),
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_holding_days": 0.0,
                "avg_trade_return": 0.0,
                "max_single_trade_loss": 0.0,
                "consecutive_loss_count": 0,
                "turnover_rate": 0.0,
                "gap_stop_count": 0,
                "gap_stop_share": 0.0,
                "benchmark_excess_return": float("nan"),
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
        drawdown_recovery_days = self._drawdown_recovery_days(curve, drawdown)

        daily_returns = curve["equity"].astype(float).pct_change().dropna()
        annual_volatility = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        if len(daily_returns) >= 2:
            daily_std = float(daily_returns.std(ddof=1))
            annual_volatility = daily_std * float(np.sqrt(TRADING_DAYS_PER_YEAR))
            mean_daily = float(daily_returns.mean())
            if daily_std > 0:
                sharpe_ratio = mean_daily / daily_std * float(np.sqrt(TRADING_DAYS_PER_YEAR))
            downside = daily_returns[daily_returns < 0]
            downside_std = float(downside.std(ddof=1)) if len(downside) >= 2 else 0.0
            if downside_std > 0:
                sortino_ratio = mean_daily / downside_std * float(np.sqrt(TRADING_DAYS_PER_YEAR))

        sells = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
        gap_stop_count = 0
        gap_stop_share = 0.0
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
            if "gap_exit" in sells.columns:
                gap_exits = sells["gap_exit"].fillna(False).astype(bool)
                stop_sells = sells["reason"].astype(str).str.startswith("stop_loss")
                gap_stop_count = int((gap_exits & stop_sells).sum())
                stop_count = int(stop_sells.sum())
                gap_stop_share = float(gap_stop_count / stop_count) if stop_count else 0.0
        else:
            win_rate = profit_loss_ratio = avg_holding_days = avg_trade_return = max_single_trade_loss = 0.0
            consecutive_loss_count = 0

        turnover = 0.0
        if not trades.empty and start_equity:
            turnover = float(pd.to_numeric(trades["amount"], errors="coerce").abs().sum() / start_equity)

        # Without a benchmark there is no excess return; NaN avoids passing off absolute return as alpha.
        benchmark_excess_return = float("nan")
        if benchmark_curve is not None and not benchmark_curve.empty and "equity" in benchmark_curve.columns:
            benchmark_return = float(benchmark_curve["equity"].iloc[-1] / benchmark_curve["equity"].iloc[0] - 1)
            benchmark_excess_return = round(total_return - benchmark_return, 6)

        return {
            "total_return": round(float(total_return), 6),
            "annual_return": round(float(annual_return), 6),
            "annual_volatility": round(annual_volatility, 6),
            "sharpe_ratio": round(sharpe_ratio, 6),
            "sortino_ratio": round(sortino_ratio, 6),
            "max_drawdown": round(max_drawdown, 6),
            "drawdown_recovery_days": drawdown_recovery_days,
            "win_rate": round(win_rate, 6),
            "profit_loss_ratio": round(profit_loss_ratio, 6) if np.isfinite(profit_loss_ratio) else float("inf"),
            "avg_holding_days": round(avg_holding_days, 3),
            "avg_trade_return": round(avg_trade_return, 6),
            "max_single_trade_loss": round(max_single_trade_loss, 2),
            "consecutive_loss_count": int(consecutive_loss_count),
            "turnover_rate": round(turnover, 6),
            "gap_stop_count": gap_stop_count,
            "gap_stop_share": round(gap_stop_share, 6),
            "benchmark_excess_return": benchmark_excess_return,
        }

    @staticmethod
    def _drawdown_recovery_days(curve: pd.DataFrame, drawdown: pd.Series) -> float:
        """Return calendar days from the max-drawdown trough back to the prior peak, NaN if unrecovered."""

        if drawdown.empty or float(drawdown.min()) >= 0:
            return 0.0
        trough_position = int(drawdown.reset_index(drop=True).idxmin())
        trough_equity_peak = float(curve["equity"].iloc[: trough_position + 1].max())
        after = curve.iloc[trough_position:]
        recovered = after[after["equity"] >= trough_equity_peak]
        if recovered.empty:
            return float("nan")
        delta = recovered["trade_date"].iloc[0] - curve["trade_date"].iloc[trough_position]
        return float(delta.days)

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
