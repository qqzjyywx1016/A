#!/usr/bin/env python3
"""Auto-detect the data gap, backfill recent baostock bars, then run selection.

Daily routine: read how far data/processed/daily_bars.parquet currently goes,
fetch the missing window up to the last trading day, refresh index and sector
aggregates, then run selection for the new latest trading day.

qfq caveat (important): baostock front-adjusted prices are re-based to the latest
date, so a dividend/split inside the gap shifts the WHOLE history. Incremental
mode therefore re-fetches the last ``--lookback-days`` and overwrites them
(_merge_existing_daily keeps the newest row), which keeps the factor lookback
window (<=60 trading days) internally consistent at the current basis. Bars older
than that window keep their prior basis; run ``--full`` periodically to re-fetch
the entire history for backtest-grade consistency.
"""

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
from astock_quant.data.storage import StorageManager
from scripts import ingest_baostock as ingest
from scripts.run_selection import compact_selection, run_selection


def existing_data_range(storage: StorageManager) -> tuple[pd.Timestamp | None, pd.Timestamp | None, list[str]]:
    """Return (min_date, max_date, standard_codes) of the local daily_bars, or Nones."""

    path = storage.processed_path / "daily_bars.parquet"
    if not Path(path).exists():
        return None, None, []
    frame = pd.read_parquet(path, columns=["stock_code", "trade_date"])
    if frame.empty:
        return None, None, []
    dates = pd.to_datetime(frame["trade_date"]).dt.normalize()
    codes = sorted(frame["stock_code"].dropna().astype(str).unique())
    return dates.min(), dates.max(), codes


def compute_fetch_window(
    data_min: pd.Timestamp,
    data_max: pd.Timestamp,
    today: pd.Timestamp,
    *,
    lookback_days: int,
    full: bool,
) -> tuple[str, str]:
    """Return (start, end) for the refetch. Full re-fetches all history; otherwise
    a recent window long enough to keep the factor lookback consistent under qfq."""

    end = today.date().isoformat()
    if full:
        return data_min.date().isoformat(), end
    start = (data_max - timedelta(days=max(int(lookback_days), 1))).date().isoformat()
    return start, end


def run(args: argparse.Namespace) -> int:
    import socket

    import baostock as bs  # type: ignore[import-not-found]

    config = load_config(args.config)
    storage = StorageManager(config)
    data_min, data_max, existing_codes = existing_data_range(storage)
    if data_max is None:
        print("no existing daily_bars.parquet; run a full `scripts/ingest_baostock.py` first")
        return 1

    today = pd.Timestamp.today().normalize()
    print(f"local data: {data_min.date()} -> {data_max.date()}; today is {today.date()}")
    if not args.full and data_max >= today:
        print("data already current; nothing to backfill")
        if not args.no_run:
            _run_selection_for(storage, args)
        return 0

    fetch_start, fetch_end = compute_fetch_window(
        data_min, data_max, today, lookback_days=args.lookback_days, full=args.full
    )
    print(f"fetching {'FULL history' if args.full else 'incremental window'} {fetch_start} -> {fetch_end}")

    socket.setdefaulttimeout(args.timeout)
    throttle = ingest.Throttle(
        sleep_seconds=args.sleep, batch_size=args.batch_size, batch_rest_seconds=args.batch_rest, jitter=args.jitter
    )
    failed: list[str] = []
    ingest._login_with_retry(bs)
    try:
        if args.full:
            # Re-enumerate the universe so new listings are picked up.
            raw_basic = ingest._fetch_stock_basic(
                bs, True, [], None, start_date=fetch_start, end_date=fetch_end, failed_codes=failed, throttle=throttle
            )
            stock_basic = ingest.normalize_stock_basic(raw_basic)
            storage.save_parquet(stock_basic, "stock_basic.parquet")
            print(f"stock_basic rows={len(stock_basic)}")
            daily_codes = raw_basic["code"].tolist()
        else:
            # Top-up only the codes we already track; new listings wait for --full.
            daily_codes = [ingest.standard_to_baostock_code(code) for code in existing_codes]

        existing_daily = ingest._read_existing_parquet(storage, "daily_bars.parquet")
        daily_bars = ingest._ingest_daily_bars_incremental(
            bs,
            storage,
            daily_codes,
            existing_daily,
            fetch_start,
            fetch_end,
            args.save_every,
            failed,
            throttle=throttle,
            relogin_every=args.relogin_every,
        )
        print(f"daily_bars rows={len(daily_bars)}")

        _update_index(bs, storage, fetch_start, fetch_end)
        sector_map = ingest._read_existing_parquet(storage, "sector_map.parquet")
        if not sector_map.empty:
            sector_daily = ingest.build_sector_daily(daily_bars, sector_map)
            storage.save_parquet(sector_daily, "sector_daily.parquet")
            print(f"sector_daily rows={len(sector_daily)}")
    finally:
        bs.logout()

    if failed:
        print(f"failed {len(failed)} codes: {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")
    if not args.no_run:
        _run_selection_for(storage, args)
    return 0


def _update_index(bs, storage: StorageManager, fetch_start: str, fetch_end: str) -> None:
    fresh = ingest._fetch_index_bars(bs, fetch_start, fetch_end)
    existing = ingest._read_existing_parquet(storage, "index_bars.parquet")
    frames = [frame for frame in [existing, fresh] if not frame.empty]
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates(["index_code", "trade_date"], keep="last").sort_values(["index_code", "trade_date"])
    storage.save_parquet(merged.reset_index(drop=True), "index_bars.parquet")
    print(f"index_bars rows={len(merged)}")


def _run_selection_for(storage: StorageManager, args: argparse.Namespace) -> None:
    _, data_max, _ = existing_data_range(storage)
    if data_max is None:
        print("no data to run selection on")
        return
    trade_date = data_max.date().isoformat()
    print(f"\nrunning selection for latest trading day {trade_date} ...")
    selected = run_selection(trade_date)
    if selected.empty:
        print("No candidates.")
        return
    csv_path = storage.result_path / f"{trade_date}_selection.csv"
    core = int((selected["rating"] == "A").sum()) if "rating" in selected.columns else 0
    print(f"\n候选 {len(selected)} 只 (A 级核心 {core} 只) — 完整字段见 {csv_path}\n")
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print(compact_selection(selected).to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill baostock data to today, then run selection")
    parser.add_argument("--config", default=None, help="Config path")
    parser.add_argument("--lookback-days", type=int, default=90, help="Incremental refetch window (keeps factor window qfq-consistent)")
    parser.add_argument("--full", action="store_true", help="Re-fetch the entire history (backtest-grade qfq consistency, slow)")
    parser.add_argument("--no-run", action="store_true", help="Only backfill data; do not run selection")
    parser.add_argument("--timeout", type=float, default=30, help="Socket timeout seconds for baostock calls")
    parser.add_argument("--save-every", type=int, default=50, help="Persist daily bars after this many successful stocks")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pause between per-stock requests")
    parser.add_argument("--batch-size", type=int, default=0, help="Take a longer rest after this many requests")
    parser.add_argument("--batch-rest", type=float, default=0.0, help="Seconds to rest between batches")
    parser.add_argument("--jitter", type=float, default=0.2, help="Randomize pauses by +-this fraction")
    parser.add_argument("--relogin-every", type=int, default=500, help="Proactively re-login every N daily requests (0 disables)")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
