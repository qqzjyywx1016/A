#!/usr/bin/env python3
"""Generate historical selection signals for backtests."""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astock_quant.config.loader import load_config
from astock_quant.data.astock_data_adapter import AStockDataAdapter
from astock_quant.data.storage import StorageManager
from scripts.run_selection import run_selection


SIGNAL_COLUMNS = [
    "trade_date",
    "stock_code",
    "total_score",
    "rating",
    "rps_5",
    "rps_10",
    "rps_20",
    "rps_60",
    "rps_composite",
    "market_regime",
    "active_sector_code",
    "active_sector_name",
    "sector_rps_5",
    "sector_rps_10",
    "sector_rps_composite",
    "suggested_position",
    "target_total_position",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    config = load_config()
    storage = StorageManager(config)
    adapter = AStockDataAdapter(config, storage=storage)
    load_start = (pd.Timestamp(args.start) - timedelta(days=180)).date().isoformat()
    market_data = {
        "stock_basic": adapter.get_stock_basic(),
        "daily_bars": adapter.get_daily_bars(load_start, args.end),
        "index_bars": adapter.get_index_bars(load_start, args.end),
        "sector_map": adapter.get_sector_map(),
        "sector_daily": adapter.get_sector_daily(load_start, args.end),
        "fund_flow": adapter.get_fund_flow(load_start, args.end),
        "trading_calendar": adapter.get_trading_calendar(args.start, args.end),
        "limit_status": pd.DataFrame(),
    }
    calendar = market_data["trading_calendar"]
    if calendar.empty:
        dates = pd.bdate_range(args.start, args.end)
    else:
        calendar["trade_date"] = pd.to_datetime(calendar["trade_date"]).dt.normalize()
        if "is_open" in calendar.columns:
            calendar = calendar[calendar["is_open"].fillna(True).astype(bool)]
        dates = calendar["trade_date"].drop_duplicates().sort_values()

    frames = []
    panel_frames = []
    for trade_date in dates:
        trade_date_str = pd.Timestamp(trade_date).date().isoformat()
        details = run_selection(
            trade_date_str,
            config=config,
            save=False,
            market_data=market_data,
            return_details=True,
        )
        selected = details["selected"]
        panel = build_backtest_panel(details["scored"], market_data["daily_bars"], trade_date_str)
        if not panel.empty:
            panel_frames.append(panel)
        if not selected.empty:
            frames.append(selected)

    signals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SIGNAL_COLUMNS)
    for column in SIGNAL_COLUMNS:
        if column not in signals.columns:
            signals[column] = pd.NA
    signals = signals[SIGNAL_COLUMNS + [column for column in signals.columns if column not in SIGNAL_COLUMNS]]
    path = storage.result_path / "signals.csv"
    signals.to_csv(path, index=False, encoding="utf-8-sig")
    panel_path = storage.result_path / "backtest_panel.parquet"
    backtest_panel = pd.concat(panel_frames, ignore_index=True) if panel_frames else pd.DataFrame(columns=BACKTEST_PANEL_COLUMNS)
    backtest_panel.to_parquet(panel_path, index=False)
    print(path)


BACKTEST_PANEL_COLUMNS = [
    "stock_code",
    "trade_date",
    "ma5",
    "ma10",
    "rps_20",
    "sector_rps_5",
    "sector_rps_10",
    "high_volume_bearish",
    "high_volume_stagnation",
    "long_upper_shadow",
    "market_regime",
]


def build_backtest_panel(scored: pd.DataFrame, daily_bars: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """Build one signal-date panel used by ContinueHoldScore in historical backtests."""

    if scored.empty:
        return pd.DataFrame(columns=BACKTEST_PANEL_COLUMNS)
    signal_date = pd.Timestamp(trade_date).normalize()
    scored = scored.copy()
    scored["trade_date"] = pd.to_datetime(scored["trade_date"]).dt.normalize()
    bars = daily_bars.copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars = bars[bars["trade_date"] <= signal_date].sort_values(["stock_code", "trade_date"]).copy()
    bars["ma5"] = bars.groupby("stock_code")["close"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    bars["ma10"] = bars.groupby("stock_code")["close"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    current = bars[bars["trade_date"] == signal_date][["stock_code", "trade_date", "ma5", "ma10"]].copy()
    panel = current.merge(scored, on=["stock_code", "trade_date"], how="left")
    for column in BACKTEST_PANEL_COLUMNS:
        if column not in panel.columns:
            panel[column] = pd.NA
    return panel[BACKTEST_PANEL_COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    main()
