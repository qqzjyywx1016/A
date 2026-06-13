import pandas as pd

from astock_quant.backtest.engine import BacktestEngine


def test_backtest_buys_on_next_day_after_signal_and_takes_profit():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10.2, "low": 9.8, "close": 10, "prev_close": 9.8, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 11.2, "low": 9.9, "close": 11.1, "prev_close": 10, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 11.2, "high": 11.4, "low": 10.8, "close": 11.2, "prev_close": 11.1, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}]
    )
    engine = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": 3,
            "stop_loss": -0.05,
            "take_profit": 0.10,
        }
    )

    result = engine.run(bars, signals)
    trades = result.trades

    assert trades.iloc[0]["trade_date"] == "2026-06-05"
    assert trades.iloc[0]["side"] == "BUY"
    assert trades.iloc[1]["side"] == "SELL"
    assert trades.iloc[1]["trade_date"] == "2026-06-08"
    assert trades.iloc[1]["reason"] == "take_profit"


def test_backtest_respects_stop_loss_and_time_exit_after_three_days():
    bars = pd.DataFrame(
            [
                {"stock_code": "600002.SH", "trade_date": "2026-06-04", "open": 10, "high": 10.2, "low": 9.8, "close": 10, "prev_close": 10, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
                {"stock_code": "600002.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.1, "low": 9.8, "close": 10, "prev_close": 10, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
                {"stock_code": "600002.SH", "trade_date": "2026-06-08", "open": 9.8, "high": 10.1, "low": 9.3, "close": 9.4, "prev_close": 10, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-04", "open": 20, "high": 20.2, "low": 19.8, "close": 20, "prev_close": 20, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-05", "open": 20, "high": 20.1, "low": 19.9, "close": 20, "prev_close": 20, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-08", "open": 20, "high": 20.1, "low": 19.9, "close": 20, "prev_close": 20, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-09", "open": 20, "high": 20.1, "low": 19.9, "close": 20, "prev_close": 20, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-10", "open": 20, "high": 20.1, "low": 19.9, "close": 20, "prev_close": 20, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"},
            {"stock_code": "600003.SH", "trade_date": "2026-06-04", "total_score": 88, "rating": "A"},
        ]
    )
    engine = BacktestEngine(
        {
            "initial_cash": 200_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": 3,
            "stop_loss": -0.05,
            "take_profit": 0.10,
        }
    )

    trades = engine.run(bars, signals).trades

    assert "stop_loss" in trades["reason"].tolist()
    assert "max_holding_days" in trades["reason"].tolist()


def test_backtest_does_not_sell_on_entry_day_when_intraday_take_profit_hits():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 12, "low": 9.8, "close": 11.8, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 11, "high": 12, "low": 10.8, "close": 11.5, "prev_close": 11.8, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}]
    )

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": 3,
            "stop_loss": -0.05,
            "take_profit": 0.10,
        }
    ).run(bars, signals).trades

    assert trades.iloc[0]["trade_date"] == "2026-06-05"
    assert trades.iloc[1]["trade_date"] == "2026-06-08"


def test_backtest_uses_intraday_stop_loss_price_after_entry_day():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.2, "low": 9.8, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 9.8, "high": 10.0, "low": 9.4, "close": 9.9, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}]
    )

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": 3,
            "stop_loss": -0.05,
            "take_profit": 0.10,
        }
    ).run(bars, signals).trades

    sell = trades[trades["side"] == "SELL"].iloc[0]
    assert sell["reason"] == "stop_loss"
    assert sell["price"] == 9.5


def test_backtest_respects_risk_off_and_participation_limits():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.5, "low": 9.8, "close": 10.2, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600002.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.5, "low": 9.8, "close": 10.2, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A", "market_regime": "risk_off"},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A", "market_regime": "strong"},
        ]
    )

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_positions": 5,
            "single_position_pct": 0.20,
            "max_participation": 0.05,
        }
    ).run(bars, signals).trades

    buys = trades[trades["side"] == "BUY"]
    assert buys["stock_code"].tolist() == ["600002.SH"]
    assert buys.iloc[0]["amount"] <= 50_000


def test_backtest_trailing_stop_uses_peak_close_after_entry_day():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 11, "low": 9.8, "close": 11, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 11.5, "high": 12.2, "low": 11.4, "close": 12, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-09", "open": 11.5, "high": 11.6, "low": 10.7, "close": 10.9, "turnover_amount": 1_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}]
    )

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": None,
            "stop_loss": -0.05,
            "take_profit": 0.30,
            "trail_pct": 0.08,
        }
    ).run(bars, signals).trades

    sell = trades[trades["side"] == "SELL"].iloc[0]
    assert sell["trade_date"] == "2026-06-09"
    assert sell["reason"] == "trailing_stop"
    assert sell["price"] == 10.9


def test_backtest_trend_break_uses_configured_ma_period():
    dates = pd.bdate_range("2026-06-01", periods=13)
    rows = []
    for index, trade_date in enumerate(dates):
        close = 10.0
        if index == 10:
            close = 10.5
        if index == 11:
            close = 9.6
        rows.append(
            {
                "stock_code": "600001.SH",
                "trade_date": trade_date,
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "turnover_amount": 1_000_000,
                "is_limit_up": False,
                "is_limit_down": False,
                "is_suspended": False,
            }
        )
    bars = pd.DataFrame(rows)
    signal_date = dates[9].strftime("%Y-%m-%d")
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": signal_date, "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_holding_days": None,
            "stop_loss": -0.20,
            "take_profit": 0.50,
            "ma_exit_period": 10,
        }
    ).run(bars, signals).trades

    sell = trades[trades["side"] == "SELL"].iloc[0]
    assert sell["reason"] == "trend_break"


def test_backtest_limits_positions_per_active_sector():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.2, "low": 9.8, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600002.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.2, "low": 9.8, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.2, "low": 9.8, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 93, "rating": "A", "market_regime": "strong", "active_sector_code": "BK001"},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "total_score": 92, "rating": "A", "market_regime": "strong", "active_sector_code": "BK001"},
            {"stock_code": "600003.SH", "trade_date": "2026-06-04", "total_score": 91, "rating": "A", "market_regime": "strong", "active_sector_code": "BK001"},
        ]
    )

    trades = BacktestEngine(
        {
            "initial_cash": 1_000_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_positions": 5,
            "max_per_sector": 2,
            "single_position_pct": 0.10,
            "max_participation": 1.0,
        }
    ).run(bars, signals).trades

    buys = trades[trades["side"] == "BUY"]
    assert buys["stock_code"].tolist() == ["600001.SH", "600002.SH"]


def test_backtest_limits_active_sector_total_exposure():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600002.SH", "trade_date": "2026-06-05", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"stock_code": "600003.SH", "trade_date": "2026-06-05", "open": 10, "high": 10, "low": 10, "close": 10, "turnover_amount": 10_000_000, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 93, "rating": "A", "market_regime": "strong", "active_sector_code": "BK001"},
            {"stock_code": "600002.SH", "trade_date": "2026-06-04", "total_score": 92, "rating": "A", "market_regime": "strong", "active_sector_code": "BK001"},
            {"stock_code": "600003.SH", "trade_date": "2026-06-04", "total_score": 91, "rating": "A", "market_regime": "strong", "active_sector_code": "BK001"},
        ]
    )

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "max_positions": 5,
            "max_per_sector": 5,
            "max_sector_exposure": 0.40,
            "single_position_pct": 0.30,
            "max_participation": 1.0,
        }
    ).run(bars, signals).trades

    buys = trades[trades["side"] == "BUY"]
    assert buys["amount"].sum() <= 40_000
    assert buys["stock_code"].tolist() == ["600001.SH", "600002.SH"]


def test_backtest_cannot_buy_sealed_limit_up():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 11, "high": 11, "low": 11, "close": 11, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": True, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "single_position_pct": 0.20,
            "max_participation": 1.0,
        }
    ).run(bars, signals).trades

    assert trades.empty


def test_backtest_buys_opened_limit_up_no_higher_than_limit_price():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 11.2, "high": 11.2, "low": 10.8, "close": 11, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_up": True, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0.02,
            "single_position_pct": 0.20,
            "max_participation": 1.0,
        }
    ).run(bars, signals).trades

    buy = trades[trades["side"] == "BUY"].iloc[0]
    assert buy["price"] == 11.0
    assert buy["limit_blocked"] is False


def test_backtest_defers_exit_on_sealed_limit_down_and_sells_later():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.1, "low": 9.9, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 9, "high": 9, "low": 9, "close": 9, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_down": True, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-09", "open": 8.1, "high": 8.1, "low": 8.1, "close": 8.1, "prev_close": 9, "turnover_amount": 1_000_000, "is_limit_down": True, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-10", "open": 7.8, "high": 8.2, "low": 7.7, "close": 8.0, "prev_close": 8.1, "turnover_amount": 1_000_000, "is_limit_down": False, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0,
            "single_position_pct": 0.20,
            "max_participation": 1.0,
            "stop_loss": -0.05,
            "take_profit": 0.50,
            "max_holding_days": None,
        }
    ).run(bars, signals).trades

    sells = trades[trades["side"] == "SELL"]
    assert len(sells) == 1
    sell = sells.iloc[0]
    assert sell["trade_date"] == "2026-06-10"
    assert sell["reason"] == "stop_loss_deferred"
    assert sell["price"] == 7.8
    assert sell["deferred_days"] == 2
    assert sell["limit_blocked"] is True


def test_backtest_opened_limit_down_sell_respects_limit_down_floor():
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-04", "open": 10, "high": 10, "low": 10, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-05", "open": 10, "high": 10.1, "low": 9.9, "close": 10, "prev_close": 10, "turnover_amount": 1_000_000, "is_suspended": False},
            {"stock_code": "600001.SH", "trade_date": "2026-06-08", "open": 8.9, "high": 9.2, "low": 9.0, "close": 9.1, "prev_close": 10, "turnover_amount": 1_000_000, "is_limit_down": True, "is_suspended": False},
        ]
    )
    signals = pd.DataFrame([{"stock_code": "600001.SH", "trade_date": "2026-06-04", "total_score": 90, "rating": "A"}])

    trades = BacktestEngine(
        {
            "initial_cash": 100_000,
            "commission_rate": 0,
            "stamp_tax_rate": 0,
            "slippage_rate": 0.02,
            "single_position_pct": 0.20,
            "max_participation": 1.0,
            "stop_loss": -0.05,
            "take_profit": 0.50,
            "max_holding_days": None,
        }
    ).run(bars, signals).trades

    sell = trades[trades["side"] == "SELL"].iloc[0]
    assert sell["reason"] == "stop_loss"
    assert sell["price"] == 9.0
