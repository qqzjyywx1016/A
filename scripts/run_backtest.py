#!/usr/bin/env python3
"""Run historical backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astock_quant.backtest.engine import BacktestEngine
from astock_quant.config.loader import load_config
from astock_quant.data.astock_data_adapter import AStockDataAdapter
from astock_quant.data.storage import StorageManager


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    config = load_config()
    storage = StorageManager(config)
    adapter = AStockDataAdapter(config, storage=storage)
    bars = adapter.get_daily_bars(args.start, args.end)
    panel_path = storage.result_path / "backtest_panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        if not panel.empty:
            bars = merge_backtest_panel(bars, panel)
    signal_path = storage.result_path / "signals.csv"
    if not signal_path.exists():
        raise FileNotFoundError(
            f"{signal_path} not found. Run `python scripts/run_batch_signals.py --start {args.start} --end {args.end}` first."
        )
    signals = pd.read_csv(signal_path)
    benchmark_curve = build_benchmark_curve(adapter.get_index_bars(args.start, args.end), config.get("backtest", {}))
    backtest_config = dict(config.get("backtest", {}))
    backtest_config.setdefault("continue_hold", config.get("continue_hold", {}))
    result = BacktestEngine(backtest_config).run(bars, signals, benchmark_curve=benchmark_curve)
    storage.save_backtest_trades(result.trades)
    storage.save_backtest_equity(result.equity_curve)
    print(result.metrics)


def merge_backtest_panel(bars: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Left join optional backtest panel columns onto daily bars."""

    if bars.empty or panel.empty:
        return bars
    bars_data = bars.copy()
    panel_data = panel.copy()
    bars_data["trade_date"] = pd.to_datetime(bars_data["trade_date"]).dt.normalize()
    panel_data["trade_date"] = pd.to_datetime(panel_data["trade_date"]).dt.normalize()
    extra_columns = [column for column in panel_data.columns if column not in {"stock_code", "trade_date"}]
    return bars_data.merge(panel_data[["stock_code", "trade_date", *extra_columns]], on=["stock_code", "trade_date"], how="left")


def build_benchmark_curve(index_bars: pd.DataFrame, backtest_config: dict) -> pd.DataFrame | None:
    """Build buy-and-hold benchmark equity from the first available index series."""

    if index_bars.empty or "close" not in index_bars.columns:
        return None
    data = index_bars.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    if "index_code" in data.columns:
        preferred = backtest_config.get("benchmark_index", "000300.SH")
        if preferred in set(data["index_code"]):
            data = data[data["index_code"] == preferred].copy()
        else:
            first_code = data["index_code"].dropna().iloc[0]
            data = data[data["index_code"] == first_code].copy()
    data = data.sort_values("trade_date")
    if data.empty:
        return None
    first_close = float(data["close"].iloc[0])
    if pd.isna(first_close) or first_close <= 0:
        return None
    initial_cash = float(backtest_config.get("initial_cash", 1_000_000))
    equity = data["close"].astype(float) / first_close * initial_cash
    return pd.DataFrame(
        {
            "trade_date": data["trade_date"].dt.date.astype(str),
            "equity": equity,
        }
    )


if __name__ == "__main__":
    main()
