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


EPS = 1e-6
# Relative tolerance for limit-price checks. qfq-adjusted prices lose the exact
# cent rounding of real limit prices, so absolute comparisons are unreliable.
LIMIT_RELATIVE_TOLERANCE = 0.003


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
        max_portfolio_drawdown = config.get("max_portfolio_drawdown")
        self.max_portfolio_drawdown = None if max_portfolio_drawdown is None else float(max_portfolio_drawdown)
        buy_rules_config = config.get("buy_rules", {})
        avoid_gap_up_pct = buy_rules_config.get("avoid_gap_up_pct") if isinstance(buy_rules_config, dict) else None
        self.avoid_gap_up_pct = None if avoid_gap_up_pct is None else float(avoid_gap_up_pct)
        continue_hold_config = config.get("continue_hold", {})
        self.continue_hold_enabled = isinstance(continue_hold_config, dict) and continue_hold_config.get("enabled", False)
        self.continue_hold_scorer = ContinueHoldScorer(continue_hold_config if isinstance(continue_hold_config, dict) else {})
        self.sell_rules = SellRuleEngine(config)
        self.broker = Broker(
            commission_rate=float(config.get("commission_rate", 0.0003)),
            stamp_tax_rate=float(config.get("stamp_tax_rate", 0.0005)),
            slippage_rate=float(config.get("slippage_rate", 0.001)),
            min_commission=float(config.get("min_commission", 0.0)),
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
        bars = self._ensure_derived_columns(bars)
        signals = signals.copy()
        if not signals.empty:
            signals["trade_date"] = pd.to_datetime(signals["trade_date"]).dt.normalize()
            signals = signals.sort_values(["trade_date", "total_score"], ascending=[True, False])

        trading_dates = sorted(bars["trade_date"].unique())
        cash = self.initial_cash
        peak_equity = self.initial_cash
        positions: dict[str, dict[str, Any]] = {}
        trades: list[Trade] = []
        equity_rows: list[dict[str, Any]] = []

        for current_date in trading_dates:
            day_bars = bars[bars["trade_date"] == current_date].set_index("stock_code")
            self._update_peak_closes(day_bars, positions)
            # Exits run before entries: A-share sell proceeds are available for buying the same day.
            cash = self._process_exits(day_bars, cash, positions, trades, trading_dates, current_date)
            previous_date = previous_trading_date(trading_dates, current_date)
            if previous_date is not None and not self._entries_blocked(day_bars, cash, positions, peak_equity):
                cash = self._process_entries(signals, previous_date, day_bars, cash, positions, trades, current_date)
            market_value = self._mark_to_market(day_bars, positions)
            equity = cash + market_value
            peak_equity = max(peak_equity, equity)
            equity_rows.append(
                {
                    "trade_date": pd.Timestamp(current_date).date().isoformat(),
                    "cash": round(cash, 2),
                    "market_value": round(market_value, 2),
                    "equity": round(equity, 2),
                }
            )

        trades_df = pd.DataFrame([asdict(trade) for trade in trades], columns=Trade.__dataclass_fields__.keys())
        for flag_column in ["limit_blocked", "gap_exit"]:
            if flag_column in trades_df.columns:
                trades_df[flag_column] = trades_df[flag_column].astype(object)
        equity_df = pd.DataFrame(equity_rows)
        metrics = MetricsCalculator().calculate(equity_df, trades_df, benchmark_curve=benchmark_curve)
        return BacktestResult(trades=trades_df, equity_curve=equity_df, metrics=metrics)

    def _ensure_derived_columns(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute prev_close and required MAs, filling NaN instead of skipping.

        Sparse panel merges leave these columns present but mostly NaN; trusting
        column existence alone silently disables exit logic on uncovered rows.
        """

        grouped = bars.groupby("stock_code")
        computed_prev_close = grouped["close"].shift(1)
        if "prev_close" in bars.columns:
            bars["prev_close"] = pd.to_numeric(bars["prev_close"], errors="coerce").fillna(computed_prev_close)
        else:
            bars["prev_close"] = computed_prev_close

        ma_periods = {5}
        if self.continue_hold_enabled:
            ma_periods.add(10)
        if self.sell_rules.ma_exit_period is not None:
            ma_periods.add(int(self.sell_rules.ma_exit_period))
        for period in sorted(ma_periods):
            column = f"ma{period}"
            computed = grouped["close"].transform(lambda s, p=period: s.rolling(p, min_periods=1).mean())
            if column in bars.columns:
                bars[column] = pd.to_numeric(bars[column], errors="coerce").fillna(computed)
            else:
                bars[column] = computed
        return bars

    def _entries_blocked(
        self,
        day_bars: pd.DataFrame,
        cash: float,
        positions: dict[str, dict[str, Any]],
        peak_equity: float,
    ) -> bool:
        """Portfolio circuit breaker: stop opening positions while in a deep drawdown."""

        if self.max_portfolio_drawdown is None or peak_equity <= 0:
            return False
        current_equity = cash + self._mark_to_market(day_bars, positions)
        drawdown = 1 - current_equity / peak_equity
        return drawdown >= self.max_portfolio_drawdown

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
            if self._is_untradable_for_buy(code, bar):
                continue
            if self._is_excessive_gap_up(bar):
                continue
            buy_price = self._buy_execution_price(code, bar)
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
            position_pct = self._position_pct_for_signal(signal)
            target_value = min(
                cash,
                current_equity * position_pct,
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
                "pending_exit_reason": None,
                "unfillable_exit_days": 0,
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

    def _position_pct_for_signal(self, signal: pd.Series) -> float:
        """Use the rating-based suggested position when present so sizing advice is actually backtested."""

        suggested = self._to_float(signal.get("suggested_position"))
        if suggested is not None and 0 < suggested <= 1:
            return suggested
        return self.single_position_pct

    def _is_excessive_gap_up(self, bar: pd.Series) -> bool:
        if self.avoid_gap_up_pct is None:
            return False
        prev_close = self._to_float(bar.get("prev_close"))
        if prev_close is None or prev_close <= 0:
            return False
        return float(bar["open"]) > prev_close * (1 + self.avoid_gap_up_pct)

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
            if self._is_suspended(bar):
                continue
            position = positions[code]
            if pd.Timestamp(position["entry_date"]).normalize() == pd.Timestamp(current_date).normalize():
                continue
            holding_days = trading_days_between(trading_dates, position["entry_date"], current_date)
            decision = self._pending_exit_decision(position, bar)
            is_deferred_execution = decision is not None
            if decision is None:
                decision = self._exit_decision(position, bar, holding_days)
            if decision is None:
                continue
            reason, base_sell_price = decision
            if self._is_sealed_limit_down(code, bar):
                position["pending_exit_reason"] = reason
                position["unfillable_exit_days"] = int(position.get("unfillable_exit_days", 0)) + 1
                continue
            deferred_days = int(position.get("unfillable_exit_days", 0)) if is_deferred_execution else 0
            trade_reason = f"{reason}_deferred" if is_deferred_execution else reason
            gap_exit = False
            if str(reason).startswith("stop_loss"):
                stop_price = float(position["entry_price"]) * (1 + self.stop_loss)
                gap_exit = base_sell_price < stop_price - EPS
            sell_price = self._sell_execution_price(code, bar, base_sell_price)
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
                    reason=trade_reason,
                    pnl=round(pnl, 2),
                    holding_days=holding_days,
                    limit_blocked=is_deferred_execution,
                    deferred_days=deferred_days,
                    gap_exit=gap_exit,
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
        if self._truthy(bar.get("is_major_event", False)) or self._truthy(bar.get("is_restructuring", False)):
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

    def _pending_exit_decision(self, position: dict[str, Any], bar: pd.Series) -> tuple[str, float] | None:
        reason = position.get("pending_exit_reason")
        if not reason:
            return None
        base_price = float(bar["open"]) if str(reason) in {"stop_loss", "trailing_stop"} else float(bar["close"])
        return str(reason), base_price

    def _is_untradable_for_buy(self, code: str, bar: pd.Series) -> bool:
        return self._is_suspended(bar) or self._is_sealed_limit_up(code, bar)

    def _buy_execution_price(self, code: str, bar: pd.Series) -> float:
        limit_up_price, _ = self._limit_prices(code, bar)
        return min(self.broker.buy_price(float(bar["open"])), limit_up_price)

    def _sell_execution_price(self, code: str, bar: pd.Series, base_price: float) -> float:
        _, limit_down_price = self._limit_prices(code, bar)
        sell_price = self.broker.sell_price(base_price)
        if self._is_limit_down(code, bar):
            return max(sell_price, limit_down_price)
        return sell_price

    def _is_sealed_limit_up(self, code: str, bar: pd.Series) -> bool:
        explicit = bar.get("is_sealed_limit_up")
        if explicit is not None and pd.notna(explicit):
            return self._truthy(explicit)
        limit_up_price, _ = self._limit_prices(code, bar)
        return self._is_limit_up(code, bar) and float(bar.get("low", 0)) >= limit_up_price * (1 - LIMIT_RELATIVE_TOLERANCE)

    def _is_sealed_limit_down(self, code: str, bar: pd.Series) -> bool:
        explicit = bar.get("is_sealed_limit_down")
        if explicit is not None and pd.notna(explicit):
            return self._truthy(explicit)
        _, limit_down_price = self._limit_prices(code, bar)
        return self._is_limit_down(code, bar) and float(bar.get("high", 0)) <= limit_down_price * (1 + LIMIT_RELATIVE_TOLERANCE)

    def _is_limit_up(self, code: str, bar: pd.Series) -> bool:
        value = bar.get("is_limit_up", False)
        if self._truthy(value):
            return True
        limit_up_price, _ = self._limit_prices(code, bar)
        close = self._to_float(bar.get("close"))
        if close is None or limit_up_price == float("inf"):
            return False
        return close >= limit_up_price * (1 - LIMIT_RELATIVE_TOLERANCE)

    def _is_limit_down(self, code: str, bar: pd.Series) -> bool:
        value = bar.get("is_limit_down", False)
        if self._truthy(value):
            return True
        _, limit_down_price = self._limit_prices(code, bar)
        close = self._to_float(bar.get("close"))
        if close is None or limit_down_price <= 0:
            return False
        return close <= limit_down_price * (1 + LIMIT_RELATIVE_TOLERANCE)

    @staticmethod
    def _is_suspended(bar: pd.Series) -> bool:
        return BacktestEngine._truthy(bar.get("is_suspended", False))

    def _limit_prices(self, code: str, bar: pd.Series) -> tuple[float, float]:
        explicit_up = self._to_float(bar.get("limit_up_price"))
        explicit_down = self._to_float(bar.get("limit_down_price"))
        if explicit_up is not None and explicit_down is not None:
            return explicit_up, explicit_down
        prev_close = self._to_float(bar.get("prev_close"))
        if prev_close is None:
            prev_close = self._to_float(bar.get("pre_close"))
        if prev_close is None or prev_close <= 0:
            return (
                explicit_up if explicit_up is not None else float("inf"),
                explicit_down if explicit_down is not None else 0.0,
            )
        limit_pct = self._limit_pct(code, bar)
        # qfq prices do not preserve exchange cent rounding, so derived limit
        # prices stay unrounded and comparisons use a relative tolerance.
        return (
            explicit_up if explicit_up is not None else prev_close * (1 + limit_pct),
            explicit_down if explicit_down is not None else prev_close * (1 - limit_pct),
        )

    @staticmethod
    def _limit_pct(code: str, bar: pd.Series) -> float:
        is_st = bar.get("is_st", False)
        if BacktestEngine._truthy(is_st):
            return 0.05
        prefix = str(code).split(".")[0]
        if prefix.startswith("30") or prefix.startswith("688"):
            return 0.20
        if prefix.startswith("8") or prefix.startswith("4"):
            return 0.30
        return 0.10

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _truthy(value: object) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "是"}:
                return True
            if normalized in {"0", "false", "no", "n", "否", ""}:
                return False
        return bool(value)

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
