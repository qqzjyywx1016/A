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

import numpy as np
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
from astock_quant.features.sentiment import SentimentFactor
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

# Research-only long-horizon momentum variants. The production momentum_score is
# 5-20d RPS, which is NEGATIVE at 1-10d (short-term reversal). These test whether a
# genuine medium-term, skip-the-most-recent-days momentum (the academic "12-1"
# construction) earns positive IC at longer forward horizons -- the question that
# decides momentum's fate. (name, lookback_days, skip_days): return from
# t-(skip+lookback) to t-skip, ranked cross-sectionally each day.
RESEARCH_MOMENTUM_SPECS = [
    ("momentum_120_skip5", 120, 5),
    ("momentum_60_skip5", 60, 5),
    ("momentum_20_skip0", 20, 0),
]
RESEARCH_FACTOR_COLUMNS = [name for name, _, _ in RESEARCH_MOMENTUM_SPECS]
ALL_FACTOR_COLUMNS = FACTOR_SCORE_COLUMNS + RESEARCH_FACTOR_COLUMNS


def parse_horizons(values: list[int] | str) -> list[int]:
    if isinstance(values, list):
        return [int(item) for item in values]
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10, 20, 60])
    parser.add_argument(
        "--reuse-scored",
        action="store_true",
        help="Skip the multi-hour scoring loop and reuse a cached factor_scores_panel.parquet "
        "from a prior run (only recomputes forward returns / research factors / IC). "
        "Use after a full run to iterate on horizons quickly.",
    )
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
    # 280 calendar days (~190 trading days) of padding so the longest research
    # momentum lookback (120d + 5d skip) is warm from the first scored date.
    load_start = (pd.Timestamp(args.start) - timedelta(days=280)).date().isoformat()

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

    # The scoring loop is the ~4h cost; cache its output so re-runs that only change
    # horizons / research factors are fast. The cache carries the per-day regime.
    panel_path = storage.result_path / "factor_scores_panel.parquet"
    if args.reuse_scored and panel_path.exists():
        scored_all = pd.read_parquet(panel_path)
        scored_all["trade_date"] = pd.to_datetime(scored_all["trade_date"]).dt.normalize()
        print(f"reusing cached scored panel ({len(scored_all)} rows) from {panel_path}", flush=True)
    else:
        scored_frames = []
        dates = sorted(
            date
            for date in daily_bars["trade_date"].drop_duplicates()
            if pd.Timestamp(args.start).normalize() <= date <= pd.Timestamp(args.end).normalize()
        )
        total_dates = len(dates)
        regime_by_date: dict[pd.Timestamp, str] = {}
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
            # Per-date market regime (history-only, no lookahead) so IC can be split
            # by regime -- the key question is whether momentum only works in某些 regimes.
            sentiment = SentimentFactor(config.get("sentiment", {})).calculate(
                bars_slice, trade_date=trade_date_str, index_bars=index_slice
            )
            regime = (
                str(sentiment["market_regime"].iloc[0]) if not sentiment.empty and "market_regime" in sentiment.columns else "unknown"
            )
            regime_by_date[trade_date] = regime
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
            # Score with the production weights AND regime gating so total_score IC
            # reflects what live selection actually ships, not the raw blend.
            scored = ScoreEngine(
                config.get("score_weights", {}), config.get("regime_weight_multipliers", {})
            ).score(factors, stock_basic, market_regime=regime)
            if not scored.empty:
                scored_frames.append(scored[["stock_code", "trade_date", *FACTOR_SCORE_COLUMNS]].copy())

        scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
        if not scored_all.empty:
            scored_all["trade_date"] = pd.to_datetime(scored_all["trade_date"]).dt.normalize()
            scored_all["market_regime"] = scored_all["trade_date"].map(regime_by_date).fillna("unknown")
            scored_all.to_parquet(panel_path, index=False)

    if scored_all.empty:
        empty_summary = pd.DataFrame(
            columns=["factor", "horizon", "ic_mean", "ic_ir", "ic_t", "positive_share", "observations"]
        )
        empty_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        markdown_path.write_text("# Factor IC Report\n\nNo scored rows available.\n", encoding="utf-8")
        print(summary_path)
        return

    if "market_regime" not in scored_all.columns:
        scored_all["market_regime"] = "unknown"
    # Research long-horizon momentum is cheap (just daily_bars), so recompute it
    # every run -- this is where the specs get tuned without paying the scoring cost.
    research = _research_momentum(daily_bars, RESEARCH_MOMENTUM_SPECS)
    scored_all = scored_all.merge(research, on=["stock_code", "trade_date"], how="left")

    forward = _forward_returns(daily_bars, horizons)
    panel = scored_all.merge(forward, on=["stock_code", "trade_date"], how="left")
    panel["year"] = panel["trade_date"].dt.year.astype(str)

    summary = _ic_summary(panel, horizons)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_year = _ic_summary_by(panel, horizons, "year")
    by_year.to_csv(storage.result_path / "factor_ic_by_year.csv", index=False, encoding="utf-8-sig")
    by_regime = _ic_summary_by(panel, horizons, "market_regime")
    by_regime.to_csv(storage.result_path / "factor_ic_by_regime.csv", index=False, encoding="utf-8-sig")
    # Spearman == Pearson on ranks; ranking first keeps this scipy-free. Constant
    # factors (e.g. all-neutral market_cap) divide by zero variance -> NaN; mute
    # the expected numpy warning so it does not spam the multi-hour run.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = panel[ALL_FACTOR_COLUMNS].rank().corr()
    _write_markdown(markdown_path, summary, corr, args.start, args.end, horizons, by_year=by_year, by_regime=by_regime)
    print(summary_path)


def _slice_by_date(frame: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    return result[result["trade_date"] <= trade_date].copy()


def _research_momentum(daily_bars: pd.DataFrame, specs: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Per (stock, date) long-horizon momentum: return from t-(skip+lookback) to t-skip.

    Skipping the most recent ``skip`` days avoids contaminating medium-term momentum
    with the short-term reversal that dominates the 1-10d window. Returned raw; the
    per-day IC ranks it, so no 0-100 rescale is needed.
    """

    bars = daily_bars.sort_values(["stock_code", "trade_date"]).copy()
    close = bars.groupby("stock_code", group_keys=False)["close"]
    out = bars[["stock_code", "trade_date"]].copy()
    for name, lookback, skip in specs:
        out[name] = close.shift(skip) / close.shift(skip + lookback) - 1
    return out


def _forward_returns(daily_bars: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    bars = daily_bars.sort_values(["stock_code", "trade_date"]).copy()
    grouped = bars.groupby("stock_code", group_keys=False)
    for horizon in horizons:
        bars[f"forward_return_{horizon}d"] = grouped["close"].shift(-horizon) / bars["close"] - 1
    columns = ["stock_code", "trade_date"] + [f"forward_return_{horizon}d" for horizon in horizons]
    return bars[columns]


def _daily_ic(frame: pd.DataFrame, factor: str, target: str) -> pd.Series:
    """Per-day rank IC = Pearson on per-day ranks (scipy-free Spearman)."""

    return (
        frame[["trade_date", factor, target]]
        .dropna()
        .groupby("trade_date")
        .apply(lambda df: df[factor].rank().corr(df[target].rank()), include_groups=False)
        .dropna()
    )


def _ic_stats(daily_ic: pd.Series) -> dict:
    """Summarize a per-day IC series into mean/IR/t/positive-share/observations."""

    observations = int(daily_ic.count())
    mean = float(daily_ic.mean()) if observations else float("nan")
    std = float(daily_ic.std(ddof=1)) if observations > 1 else float("nan")
    return {
        "ic_mean": round(mean, 6) if pd.notna(mean) else pd.NA,
        "ic_ir": round(mean / std, 6) if pd.notna(std) and std != 0 else pd.NA,
        "ic_t": round(mean / (std / (observations**0.5)), 6) if pd.notna(std) and std != 0 and observations > 1 else pd.NA,
        "positive_share": round(float((daily_ic > 0).mean()), 6) if observations else pd.NA,
        "observations": observations,
    }


def _ic_summary(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    # Constant factors on a day have zero rank variance -> NaN IC; mute the
    # expected numpy divide warning rather than spam it once per factor per day.
    with np.errstate(invalid="ignore", divide="ignore"):
        factors = [column for column in ALL_FACTOR_COLUMNS if column in panel.columns]
        rows = [
            {"factor": factor, "horizon": horizon, **_ic_stats(_daily_ic(panel, factor, f"forward_return_{horizon}d"))}
            for horizon in horizons
            for factor in factors
        ]
    return pd.DataFrame(rows)


def _ic_summary_by(panel: pd.DataFrame, horizons: list[int], segment_col: str) -> pd.DataFrame:
    """IC summary split into segments (e.g. calendar year or market regime)."""

    with np.errstate(invalid="ignore", divide="ignore"):
        rows = []
        factors = [column for column in ALL_FACTOR_COLUMNS if column in panel.columns]
        for segment, group in panel.groupby(segment_col):
            for horizon in horizons:
                target = f"forward_return_{horizon}d"
                for factor in factors:
                    rows.append(
                        {"segment": str(segment), "factor": factor, "horizon": horizon, **_ic_stats(_daily_ic(group, factor, target))}
                    )
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without tabulate."""

    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    start: str,
    end: str,
    horizons: list[int],
    *,
    by_year: pd.DataFrame | None = None,
    by_regime: pd.DataFrame | None = None,
) -> None:
    corr_table = (
        _df_to_markdown(corr.round(4).rename_axis("factor").reset_index()) if not corr.empty else "No correlation matrix."
    )
    lines = [
        "# Factor IC Report",
        "",
        f"- Window: {start} to {end}",
        f"- Horizons: {', '.join(str(horizon) for horizon in horizons)} trading days",
        "- Scope: research-only; forward returns are used only for post-hoc IC validation.",
        "- `momentum_<L>_skip<S>`: research-only medium-term momentum = return from "
        "t-(S+L) to t-S (skips the most recent S days to dodge short-term reversal). "
        "Positive IC at h20/h60 would justify keeping a momentum book.",
        "",
        "## IC Summary (full window)",
        "",
        _df_to_markdown(summary) if not summary.empty else "No IC rows.",
        "",
    ]
    if by_year is not None and not by_year.empty:
        lines += [
            "## IC by Calendar Year",
            "",
            "Tells whether momentum is regime-dependent (positive in trending years) or broken throughout.",
            "",
            _df_to_markdown(by_year),
            "",
        ]
    if by_regime is not None and not by_regime.empty:
        lines += [
            "## IC by Market Regime",
            "",
            "Regime is the daily sentiment state (history-only). Watch whether momentum flips sign by regime.",
            "",
            _df_to_markdown(by_regime),
            "",
        ]
    lines += [
        "## Factor Score Spearman Correlation",
        "",
        corr_table,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
