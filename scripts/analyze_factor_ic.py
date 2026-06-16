#!/usr/bin/env python3
"""Analyze rank IC for stage-one factors.

This is a research-only script. Forward returns intentionally use future bars
for post-trade evaluation and are never fed back into the signal path.
"""

from __future__ import annotations

import argparse
import logging
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
from astock_quant.features.market_cap import MarketCapFactor
from astock_quant.features.momentum import MomentumFactor
from astock_quant.features.reversal import ReversalFactor
from astock_quant.features.sector import SectorFactor
from astock_quant.features.volume import VolumeFactor
from astock_quant.scoring.score_engine import ScoreEngine
from scripts.run_selection import build_market_cap_snapshot, enrich_daily_bars_for_selection, sector_factor_config


FACTOR_SCORE_COLUMNS = [
    "momentum_score",
    "volume_score",
    "sector_score",
    "market_cap_score",
    "reversal_score",
    "total_score",
]


def parse_horizons(values: list[int] | str) -> list[int]:
    if isinstance(values, list):
        return [int(item) for item in values]
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    return parser


def _quiet_warmup_logs() -> None:
    """Silence the expected per-date warmup/structural warnings during the IC sweep.

    Early dates lack enough history for RPS, and market_cap/fund_flow/pattern/
    sentiment are not part of the five scored factors, so these loggers would
    otherwise emit thousands of expected WARNINGs and bury the progress output.
    """

    for name in (
        "astock_quant.features.momentum",
        "astock_quant.features.sector",
        "astock_quant.scoring.score_engine",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    args = build_arg_parser().parse_args()
    _quiet_warmup_logs()

    config = load_config()
    storage = StorageManager(config)
    adapter = AStockDataAdapter(config, storage=storage)
    horizons = parse_horizons(args.horizons)
    load_start = (pd.Timestamp(args.start) - timedelta(days=180)).date().isoformat()

    stock_basic = adapter.get_stock_basic()
    daily_bars = adapter.get_daily_bars(load_start, args.end)
    index_bars = adapter.get_index_bars(load_start, args.end)
    sector_map = adapter.get_sector_map()
    sector_daily = adapter.get_sector_daily(load_start, args.end)

    storage.result_path.mkdir(parents=True, exist_ok=True)
    storage.report_path.mkdir(parents=True, exist_ok=True)
    summary_path = storage.result_path / "factor_ic_report.csv"
    markdown_path = storage.report_path / "factor_ic_report.md"

    if daily_bars.empty:
        empty_summary = pd.DataFrame(
            columns=["factor", "horizon", "ic_mean", "ic_ir", "ic_t", "positive_share", "observations"]
        )
        empty_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        markdown_path.write_text("# Factor IC Report\n\nNo daily bars available.\n", encoding="utf-8")
        print(summary_path)
        return

    daily_bars = enrich_daily_bars_for_selection(daily_bars)
    daily_bars["trade_date"] = pd.to_datetime(daily_bars["trade_date"]).dt.normalize()
    scored_frames = []
    dates = sorted(
        date
        for date in daily_bars["trade_date"].drop_duplicates()
        if pd.Timestamp(args.start).normalize() <= date <= pd.Timestamp(args.end).normalize()
    )
    total_dates = len(dates)
    print(f"scoring {total_dates} trading days from {args.start} to {args.end} (this can take a while)...", flush=True)
    for index, trade_date in enumerate(dates, start=1):
        trade_date_str = pd.Timestamp(trade_date).date().isoformat()
        bars_slice = daily_bars[daily_bars["trade_date"] <= trade_date].copy()
        latest = bars_slice[bars_slice["trade_date"] == trade_date].copy()
        if index % 20 == 0 or index == total_dates:
            print(f"[{index}/{total_dates}] scored through {trade_date_str}", flush=True)
        if latest.empty:
            continue
        if not stock_basic.empty:
            latest = latest.merge(stock_basic, on="stock_code", how="left", suffixes=("", "_basic"))
        sector_slice = _slice_by_date(sector_daily, trade_date)
        index_slice = _slice_by_date(index_bars, trade_date)
        sector_factor = SectorFactor(sector_factor_config(config)).calculate(
            bars_slice,
            trade_date=trade_date_str,
            sector_map=sector_map,
            sector_daily=sector_slice,
        )
        factors = {
            "momentum": MomentumFactor().calculate(
                bars_slice,
                trade_date=trade_date_str,
                index_bars=index_slice,
                sector_daily=sector_slice,
                sector_map=sector_map,
            ),
            "volume": VolumeFactor().calculate(bars_slice, trade_date=trade_date_str),
            "sector": sector_factor,
            "market_cap": MarketCapFactor().calculate(
                build_market_cap_snapshot(latest, sector_factor),
                trade_date=trade_date_str,
            ),
            "reversal": ReversalFactor().calculate(bars_slice, trade_date=trade_date_str),
        }
        scored = ScoreEngine(config.get("score_weights", {})).score(factors, stock_basic)
        if not scored.empty:
            scored_frames.append(scored[["stock_code", "trade_date", *FACTOR_SCORE_COLUMNS]].copy())

    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    if scored_all.empty:
        empty_summary = pd.DataFrame(
            columns=["factor", "horizon", "ic_mean", "ic_ir", "ic_t", "positive_share", "observations"]
        )
        empty_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        markdown_path.write_text("# Factor IC Report\n\nNo scored rows available.\n", encoding="utf-8")
        print(summary_path)
        return

    scored_all["trade_date"] = pd.to_datetime(scored_all["trade_date"]).dt.normalize()
    forward = _forward_returns(daily_bars, horizons)
    panel = scored_all.merge(forward, on=["stock_code", "trade_date"], how="left")
    summary = _ic_summary(panel, horizons)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    corr = scored_all[FACTOR_SCORE_COLUMNS].corr(method="spearman")
    _write_markdown(markdown_path, summary, corr, args.start, args.end, horizons)
    print(summary_path)


def _slice_by_date(frame: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    return result[result["trade_date"] <= trade_date].copy()


def _forward_returns(daily_bars: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    bars = daily_bars.sort_values(["stock_code", "trade_date"]).copy()
    grouped = bars.groupby("stock_code", group_keys=False)
    for horizon in horizons:
        bars[f"forward_return_{horizon}d"] = grouped["close"].shift(-horizon) / bars["close"] - 1
    columns = ["stock_code", "trade_date"] + [f"forward_return_{horizon}d" for horizon in horizons]
    return bars[columns]


def _ic_summary(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        target = f"forward_return_{horizon}d"
        for factor in FACTOR_SCORE_COLUMNS:
            daily_ic = (
                panel[["trade_date", factor, target]]
                .dropna()
                .groupby("trade_date")
                .apply(lambda df: df[factor].corr(df[target], method="spearman"), include_groups=False)
                .dropna()
            )
            observations = int(daily_ic.count())
            mean = float(daily_ic.mean()) if observations else float("nan")
            std = float(daily_ic.std(ddof=1)) if observations > 1 else float("nan")
            rows.append(
                {
                    "factor": factor,
                    "horizon": horizon,
                    "ic_mean": round(mean, 6) if pd.notna(mean) else pd.NA,
                    "ic_ir": round(mean / std, 6) if pd.notna(std) and std != 0 else pd.NA,
                    "ic_t": round(mean / (std / (observations**0.5)), 6)
                    if pd.notna(std) and std != 0 and observations > 1
                    else pd.NA,
                    "positive_share": round(float((daily_ic > 0).mean()), 6) if observations else pd.NA,
                    "observations": observations,
                }
            )
    return pd.DataFrame(rows)


def _write_markdown(
    path: Path,
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    start: str,
    end: str,
    horizons: list[int],
) -> None:
    lines = [
        "# Factor IC Report",
        "",
        f"- Window: {start} to {end}",
        f"- Horizons: {', '.join(str(horizon) for horizon in horizons)} trading days",
        "- Scope: research-only; forward returns are used only for post-hoc IC validation.",
        "",
        "## IC Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No IC rows.",
        "",
        "## Factor Score Spearman Correlation",
        "",
        corr.round(4).to_markdown() if not corr.empty else "No correlation matrix.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
