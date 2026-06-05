"""Daily-bar backtesting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from astock_quant.backtest.broker import Broker
from astock_quant.backtest.metrics import MetricsCalculator
from astock_quant.backtest.trade import Trade
from astock_quant.strategy.continue_hold import ContinueHoldScorer
from astock_quant.strategy.position import PositionSizer
from astock_quant.strategy.sell_rules import SellRuleEngine
from astock_quant.utils.calendar import previous_trading_date, trading_days_between


@dataclass(slots=True)
class BacktestResult:
    """Backtest outputs: trade records, equity curve and summary metrics."""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float | int]


class BacktestEngine:
    """Simulate T+1 execution for daily A-share momentum signals."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.initial_cash = float(config.get("initial_cash", 1_000_000))
        max_holding_days = config.get("max_holding_days", 3)
        self.max_holding_days = None if max_holding_days is None else int(max_holding_days)
        self.stop_loss = float(config.get("stop_loss", -0.05))
        self.take_profit = float(config.get("take_profit", 0.10))
        self.max_positions = int(config.get("max_positions", 5))
        self.max_per_sector = int(config.get("max_per_sector", 2))
        self.max_sector_exposure = float(config.get("max_sector_exposure", 0.40))
        self.single_position_pct = float(config.get("single_position_pct", 0.20))
        self.max_participation = float(config.get("max_participation", 0.05))
        continue_hold_config = config.get("continue_hold", {})
        self.continue_hold_enabled = isinstance(continue_hold_config, dict) and continue_hold_config.get("enabled", False)
        self.continue_hold_scorer = ContinueHoldScorer(continue_hold_config if isinstance(continue_hold_config, dict) else {})
        self.sell_rules = SellRuleEngine(config)
        self.broker = Broker(
            commission_rate=float(config.get("commission_rate", 0.0003)),
            stamp_tax_rate=float(config.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(config.get("slippage_rate", 0.001)),
        )

    def run(
        self,
        daily_bars: pd.DataFrame,
        signals: pd.DataFrame,
        *,
        benchmark_curve: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run a no-lookahead backtest using T-day signals and T+1 buys."""

        if daily_bars.empty:
            empty_trades = pd.DataFrame(columns=Trade.__dataclass_fields__.keys())
            equity = pd.DataFrame(columns=["trade_date", "cash", "market_value", "equity"])
            return BacktestResult(empty_trades, equity, MetricsCalculator().calculate(equity, empty_trades))

        bars = daily_bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        bars = bars.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
        if "ma5" not in bars.columns:
            bars["ma5"] = bars.groupby("stock_code")["close"].transform(lambda s: s.rolling(5, min_periods=1).mean())
        if self.continue_hold_enabled and "ma10" not in bars.columns:
            bars["ma10"] = bars.groupby("stock_code")["close"].transform(lambda s: s.rolling(10, min_periods=1).mean())
        if self.sell_rules.ma_exit_period is not None:
            ma_column = f"ma{self.sell_rules.ma_exit_period}"
            if ma_column not in bars.columns:
                bars[ma_column] = bars.groupby("stock_code")["close"].transform(
                    lambda s: s.rolling(self.sell_rules.ma_exit_period, min_periods=1).mean()
                )
        signals = signals.copy()
        if not signals.empty:
            signals["trade_date"] = pd.to_datetime(signals["trade_date"]).dt.normalize()
            signals = signals.sort_values(["trade_date", "total_score"], ascending=[True, False])

        trading_dates = sorted(bars["trade_date"].unique())
        cash = self.initial_cash
        positions: dict[str, dict[str, Any]] = {}
        trades: list[Trade] = []
        equity_rows: list[dict[str, Any]] = []

        for current_date in trading_dates:
            day_bars = bars[bars["trade_date"] == current_date].set_index("stock_code")
            previous_date = previous_trading_date(trading_dates, current_date)
            if previous_date is not None:
                cash = self._process_entries(signals, previous_date, day_bars, cash, positions, trades, current_date)
            self._update_peak_closes(day_bars, positions)
            cash = self._process_exits(day_bars, cash, positions, trades, trading_dates, current_date)
            market_value = self._mark_to_market(day_bars, positions)
            equity_rows.append(
                {
                    "trade_date": pd.Timestamp(current_date).date().isoformat(),
                    "cash": round(cash, 2),
                    "market_value": round(market_value, 2),
                    "equity": round(cash + market_value, 2),
                }
            )

        trades_df = pd.DataFrame([asdict(trade) for trade in trades])
        equity_df = pd.DataFrame(equity_rows)
        metrics = MetricsCalculator().calculate(equity_df, trades_df, benchmark_curve=benchmark_curve)
        return BacktestResult(trades=trades_df, equity_curve=equity_df, metrics=metrics)

    def _process_entries(
        self,
        signals: pd.DataFrame,
        signal_date: pd.Timestamp,
        day_bars: pd.DataFrame,
        cash: float,
        positions: dict[str, dict[str, Any]],
        trades: list[Trade],
        current_date: pd.Timestamp,
    ) -> float:
        if signals.empty or len(positions) >= self.max_positions:
            return cash
        signal_rows = signals[signals["trade_date"] == signal_date].sort_values("total_score", ascending=False)
        for _, signal in signal_rows.iterrows():
            if len(positions) >= self.max_positions:
                break
            market_regime = str(signal.get("market_regime") or "neutral")
            if market_regime == "risk_off":
                continue
            code = signal["stock_code"]
            if code in positions or code not in day_bars.index:
                continue
            active_sector_code = signal.get("active_sector_code")
            if pd.notna(active_sector_code):
                same_sector_count = sum(
                    1 for position in positions.values() if position.get("active_sector_code") == active_sector_code
                )
                if same_sector_count >= self.max_per_sector:
                    continue
            bar = day_bars.loc[code]
            if self._is_untradable_for_buy(bar):
                continue
            buy_price = self.broker.buy_price(float(bar["open"]))
            current_market_value = self._mark_to_market(day_bars, positions)
            current_equity = cash + current_market_value
            total_cap = current_equity * PositionSizer.REGIME_TOTAL_POSITION.get(market_regime, 0.40)
            remaining_cap = max(total_cap - current_market_value, 0.0)
            sector_remaining_cap = self._sector_remaining_cap(day_bars, positions, active_sector_code, current_equity)
            if sector_remaining_cap <= 0:
                continue
            turnover_amount = bar.get("turnover_amount")
            participation_cap = (
                float(turnover_amount) * self.max_participation
                if turnover_amount is not None and pd.notna(turnover_amount)
                else float("inf")
            )
            # Strong small-cap names can have high impact costs; cap participation to avoid unrealistic fills.
            target_value = min(
                cash,
                current_equity * self.single_position_pct,
                remaining_cap,
                sector_remaining_cap,
                participation_cap,
            )
            shares = int(target_value / buy_price / 100) * 100
            if shares <= 0:
                continue
            amount = buy_price * shares
            fee = self.broker.commission(amount)
            if cash < amount + fee:
                continue
            cash -= amount + fee
            positions[code] = {
                "stock_code": code,
                "entry_date": current_date,
                "entry_price": buy_price,
                "shares": shares,
                "cost": amount + fee,
                "peak_close": float(bar["close"]),
                "active_sector_code": active_sector_code if pd.notna(active_sector_code) else None,
            }
            trades.append(
                Trade(
                    trade_date=pd.Timestamp(current_date).date().isoformat(),
                    stock_code=code,
                    side="BUY",
                    price=round(buy_price, 4),
                    shares=shares,
                    amount=round(amount, 2),
                    fee=round(fee, 2),
                    tax=0.0,
                    reason="signal_t_plus_1",
                )
            )
        return cash

    def _process_exits(
        self,
        day_bars: pd.DataFrame,
        cash: float,
        positions: dict[str, dict[str, Any]],
        trades: list[Trade],
        trading_dates: list[pd.Timestamp],
        current_date: pd.Timestamp,
    ) -> float:
        for code in list(positions.keys()):
            if code not in day_bars.index:
                continue
            bar = day_bars.loc[code]
            if self._is_untradable_for_sell(bar):
                continue
            position = positions[code]
            if pd.Timestamp(position["entry_date"]).normalize() == pd.Timestamp(current_date).normalize():
                continue
            holding_days = trading_days_between(trading_dates, position["entry_date"], current_date)
            decision = self._exit_decision(position, bar, holding_days)
            if decision is None:
                continue
            reason, base_sell_price = decision
            sell_price = self.broker.sell_price(base_sell_price)
            shares = int(position["shares"])
            amount = sell_price * shares
            fee = self.broker.commission(amount)
            tax = self.broker.stamp_tax(amount)
            pnl = amount - fee - tax - float(position["cost"])
            cash += amount - fee - tax
            trades.append(
                Trade(
                    trade_date=pd.Timestamp(current_date).date().isoformat(),
                    stock_code=code,
                    side="SELL",
                    price=round(sell_price, 4),
                    shares=shares,
                    amount=round(amount, 2),
                    fee=round(fee, 2),
                    tax=round(tax, 2),
                    reason=reason,
                    pnl=round(pnl, 2),
                    holding_days=holding_days,
                )
            )
            del positions[code]
        return cash

    def _exit_decision(self, position: dict[str, Any], bar: pd.Series, holding_days: int) -> tuple[str, float] | None:
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        entry_price = float(position["entry_price"])
        stop_price = entry_price * (1 + self.stop_loss)
        take_price = entry_price * (1 + self.take_profit)
        if low <= stop_price:
            return "stop_loss", min(open_price, stop_price)
        if str(bar.get("market_regime", "")).lower() == "risk_off":
            return "market_risk_off", close
        if bool(bar.get("is_major_event", False)) or bool(bar.get("is_restructuring", False)):
            return "major_event_risk", close
        if self.sell_rules.trail_pct is not None:
            peak_close = float(position.get("peak_close", entry_price))
            if close <= peak_close * (1 - self.sell_rules.trail_pct):
                return "trailing_stop", close
        if high >= take_price:
            return "take_profit", max(open_price, take_price)
        if self.continue_hold_enabled and self.continue_hold_scorer.can_evaluate(bar):
            decision = self.continue_hold_scorer.evaluate(bar)
            if decision.action == "exit":
                return decision.reason or "low_continue_hold_score", close
            if self.max_holding_days is not None and holding_days >= self.max_holding_days:
                return "max_holding_days", close
            return None
        # Rule and time exits use the current close in this daily-bar backtest.
        # This keeps non-intraday rules conservative and consistent with close-known signals.
        reason = self.sell_rules.evaluate(position, bar, holding_days=holding_days)
        return (reason, close) if reason is not None else None

    @staticmethod
    def _is_untradable_for_buy(bar: pd.Series) -> bool:
        return bool(bar.get("is_suspended", False)) or bool(bar.get("is_limit_up", False))

    @staticmethod
    def _is_untradable_for_sell(bar: pd.Series) -> bool:
        return bool(bar.get("is_suspended", False)) or bool(bar.get("is_limit_down", False))

    @staticmethod
    def _mark_to_market(day_bars: pd.DataFrame, positions: dict[str, dict[str, Any]]) -> float:
        value = 0.0
        for code, position in positions.items():
            if code in day_bars.index:
                value += float(day_bars.loc[code, "close"]) * int(position["shares"])
            else:
                value += float(position["entry_price"]) * int(position["shares"])
        return value

    def _sector_remaining_cap(
        self,
        day_bars: pd.DataFrame,
        positions: dict[str, dict[str, Any]],
        active_sector_code: object,
        current_equity: float,
    ) -> float:
        if pd.isna(active_sector_code):
            return float("inf")
        sector_value = 0.0
        for code, position in positions.items():
            if position.get("active_sector_code") != active_sector_code:
                continue
            if code in day_bars.index:
                sector_value += float(day_bars.loc[code, "close"]) * int(position["shares"])
            else:
                sector_value += float(position["entry_price"]) * int(position["shares"])
        sector_cap = current_equity * self.max_sector_exposure
        return max(sector_cap - sector_value, 0.0)

    @staticmethod
    def _update_peak_closes(day_bars: pd.DataFrame, positions: dict[str, dict[str, Any]]) -> None:
        for code, position in positions.items():
            if code not in day_bars.index:
                continue
            close = float(day_bars.loc[code, "close"])
            position["peak_close"] = max(float(position.get("peak_close", close)), close)
