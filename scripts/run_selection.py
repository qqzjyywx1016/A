#!/usr/bin/env python3
"""Run daily stock selection for a specified date."""

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
from astock_quant.features.fund_flow import FundFlowFactor
from astock_quant.features.market_cap import MarketCapFactor
from astock_quant.features.momentum import MomentumFactor
from astock_quant.features.pattern import PatternFactor
from astock_quant.features.reversal import ReversalFactor
from astock_quant.features.sector import SectorFactor
from astock_quant.features.sentiment import SentimentFactor
from astock_quant.features.volume import VolumeFactor
from astock_quant.scoring.score_engine import ScoreEngine
from astock_quant.strategy.buy_rules import BuyRuleEngine
from astock_quant.strategy.overheat_filter import OverheatFilter
from astock_quant.strategy.position import PositionSizer
from astock_quant.strategy.selector import StockSelector
from astock_quant.universe.filters import UniverseFilter
from astock_quant.utils.logger import get_logger

logger = get_logger(__name__)


def selection_config(config: dict) -> dict:
    """Merge selection and RPS config for StockSelector while preserving old call shape."""

    merged = dict(config.get("selection", {}))
    merged["rps"] = config.get("rps", {})
    if "sector_rps" in config:
        merged["sector_rps"] = config.get("sector_rps", {})
    return merged


def sector_factor_config(config: dict) -> dict:
    """Merge sector RPS scoring config with backtest/live sector availability guard."""

    merged = dict(config.get("sector_rps", {}))
    merged.update(config.get("sector", {}))
    return merged


def run_selection(
    trade_date: str,
    config: dict | None = None,
    *,
    save: bool = True,
    market_data: dict[str, pd.DataFrame] | None = None,
    return_details: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Run the full rule-scoring selection pipeline."""

    config = config or load_config()
    storage = StorageManager(config)
    start_date = (pd.Timestamp(trade_date) - timedelta(days=180)).date().isoformat()
    if market_data is None:
        adapter = AStockDataAdapter(config, storage=storage)
        stock_basic = adapter.get_stock_basic()
        daily_bars = adapter.get_daily_bars(start_date, trade_date)
        sector_map = adapter.get_sector_map()
        sector_daily = adapter.get_sector_daily(start_date, trade_date)
        fund_flow = adapter.get_fund_flow(start_date, trade_date)
        index_bars = adapter.get_index_bars(start_date, trade_date)
        limit_status = adapter.get_limit_status(trade_date)
    else:
        sliced = slice_market_data(market_data, trade_date)
        stock_basic = sliced.get("stock_basic", pd.DataFrame())
        daily_bars = sliced.get("daily_bars", pd.DataFrame())
        sector_map = sliced.get("sector_map", pd.DataFrame())
        sector_daily = sliced.get("sector_daily", pd.DataFrame())
        fund_flow = sliced.get("fund_flow", pd.DataFrame())
        index_bars = sliced.get("index_bars", pd.DataFrame())
        limit_status = sliced.get("limit_status", pd.DataFrame())
    if daily_bars.empty:
        empty = pd.DataFrame()
        if save:
            storage.save_daily_selection(empty, trade_date)
        return {"selected": empty, "rejected": pd.DataFrame(), "scored": empty} if return_details else empty

    daily_bars = enrich_daily_bars_for_selection(daily_bars)
    latest = daily_bars[pd.to_datetime(daily_bars["trade_date"]).dt.normalize() == pd.Timestamp(trade_date).normalize()].copy()
    if latest.empty:
        # The signal date has no bars (weekend/holiday or beyond ingested range):
        # bail out with a clear message instead of a wall of neutral-factor warnings.
        available = pd.to_datetime(daily_bars["trade_date"]).dt.normalize()
        last_available = available.max()
        hint = last_available.date().isoformat() if pd.notna(last_available) else "unknown"
        logger.warning(
            "no market data for trade_date=%s (not a trading day, or beyond ingested range); "
            "last available trading day in data is %s. Re-run with --date %s",
            trade_date,
            hint,
            hint,
        )
        empty = pd.DataFrame()
        if save:
            storage.save_daily_selection(empty, trade_date)
        return {"selected": empty, "rejected": pd.DataFrame(), "scored": empty} if return_details else empty
    if not stock_basic.empty:
        latest = latest.merge(stock_basic, on="stock_code", how="left", suffixes=("", "_basic"))
        for column in ["stock_name", "sector"]:
            basic_column = f"{column}_basic"
            if basic_column in latest.columns:
                latest[column] = latest.get(column).fillna(latest[basic_column])
    universe = UniverseFilter(config.get("universe", {})).apply(latest, as_of_date=trade_date)
    allowed_codes = set(universe["stock_code"]) if not universe.empty else set()

    market_codes = daily_bars["stock_code"].dropna().drop_duplicates().tolist()
    sector_factor = SectorFactor(sector_factor_config(config)).calculate(
        daily_bars,
        trade_date=trade_date,
        sector_map=sector_map,
        sector_daily=sector_daily,
    )
    market_cap_input = build_market_cap_snapshot(latest, sector_factor)
    factors = {
        "momentum": MomentumFactor().calculate(
            daily_bars,
            trade_date=trade_date,
            index_bars=index_bars,
            sector_daily=sector_daily,
            sector_map=sector_map,
        ),
        "volume": VolumeFactor().calculate(daily_bars, trade_date=trade_date),
        "sector": sector_factor,
        "market_cap": MarketCapFactor().calculate(market_cap_input, trade_date=trade_date),
        "reversal": ReversalFactor().calculate(daily_bars, trade_date=trade_date),
        "fund_flow": FundFlowFactor().calculate(fund_flow, trade_date=trade_date, stock_codes=market_codes),
        "pattern": PatternFactor().calculate(daily_bars, trade_date=trade_date),
        "sentiment": SentimentFactor(config.get("sentiment", {})).calculate(
            daily_bars,
            trade_date=trade_date,
            limit_status=limit_status,
            index_bars=index_bars,
        ),
    }
    scored = ScoreEngine(config.get("score_weights", {})).score(factors, stock_basic)
    if allowed_codes:
        scored = scored[scored["stock_code"].isin(allowed_codes)].copy()
    else:
        scored = scored.iloc[0:0].copy()
    if "is_limit_up" in latest.columns and not scored.empty:
        scored = scored.merge(latest[["stock_code", "is_limit_up"]].drop_duplicates("stock_code"), on="stock_code", how="left")
    rejected = pd.DataFrame(columns=["stock_code", "reject_reason"])
    overheat_config = config.get("overheat")
    if isinstance(overheat_config, dict) and overheat_config.get("enabled", True) and not scored.empty:
        scored, rejected = OverheatFilter(overheat_config).apply(scored)
    # Rating-based sizing runs before selection so the selector's sector-exposure
    # cap sees real suggested_position values and the backtest can honor them.
    if not scored.empty and "rating" in scored.columns:
        market_regime = "neutral"
        if "market_regime" in scored.columns and not scored["market_regime"].dropna().empty:
            market_regime = str(scored["market_regime"].dropna().mode().iloc[0])
        scored = PositionSizer(config.get("position", {})).suggest(scored, market_regime=market_regime)
    selected = StockSelector(selection_config(config)).select(scored, trade_date=trade_date)["watch_pool"]
    plan_fields = [column for column in ["stock_code", "close", "high", "ma5", "ma10"] if column in latest.columns]
    if not selected.empty and len(plan_fields) > 1:
        selected = selected.merge(
            latest[plan_fields].drop_duplicates("stock_code"), on="stock_code", how="left", suffixes=("", "_bar")
        )
    selected = BuyRuleEngine(config.get("buy_rules", {})).generate(selected)
    # Concept tags are current-only (no point-in-time history), so they are
    # attached on the LIVE path only. market_data is supplied exclusively by the
    # batch/backtest caller, so gating on `market_data is None` guarantees the
    # backtest never sees today's tags (which would be look-ahead bias).
    if market_data is None and not selected.empty:
        selected = attach_concept_tags(selected, storage)
    if save:
        storage.save_daily_selection(selected, trade_date)
        storage.save_rejected_candidates(rejected, trade_date)
    if return_details:
        return {"selected": selected, "rejected": rejected, "scored": scored}
    return selected


def attach_concept_tags(selected: pd.DataFrame, storage: StorageManager) -> pd.DataFrame:
    """Left-merge current Eastmoney concept tags for live display (no-op if absent)."""

    path = storage.processed_path / "concept_map.parquet"
    if not Path(path).exists():
        return selected
    try:
        concept_map = pd.read_parquet(path)
    except Exception as exc:  # a corrupt cache must not break selection
        logger.warning("failed reading concept_map.parquet: %s", exc)
        return selected
    keep = [column for column in ["stock_code", "top_concepts", "concept_tags", "top_concept"] if column in concept_map.columns]
    if "stock_code" not in keep or len(keep) <= 1:
        return selected
    merged = selected.merge(concept_map[keep].drop_duplicates("stock_code"), on="stock_code", how="left")
    for column in keep:
        if column != "stock_code" and column in merged.columns:
            merged[column] = merged[column].fillna("")
    return merged


def enrich_daily_bars_for_selection(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Add rolling snapshot fields needed by universe and stage-one factors."""

    result = daily_bars.copy()
    if result.empty or "stock_code" not in result.columns or "trade_date" not in result.columns:
        return result
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result = result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    grouped = result.groupby("stock_code", group_keys=False)
    if "turnover_amount" in result.columns and "avg_turnover_amount_20d" not in result.columns:
        result["avg_turnover_amount_20d"] = grouped["turnover_amount"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        )
    if "turnover_rate" in result.columns and "avg_turnover_rate_20d" not in result.columns:
        result["avg_turnover_rate_20d"] = grouped["turnover_rate"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    for period in [5, 10]:
        column = f"ma{period}"
        if column not in result.columns:
            result[column] = grouped["close"].transform(lambda s, p=period: s.rolling(p, min_periods=1).mean())
    return result


def build_market_cap_snapshot(latest: pd.DataFrame, sector_factor: pd.DataFrame) -> pd.DataFrame:
    """Merge sector regime context into the stock snapshot for MarketCapFactor."""

    snapshot = latest.copy()
    if sector_factor.empty or "stock_code" not in sector_factor.columns:
        return snapshot
    keep = [
        column
        for column in ["stock_code", "active_sector_code", "active_sector_name", "sector_regime"]
        if column in sector_factor.columns
    ]
    if len(keep) <= 1:
        return snapshot
    sector_context = sector_factor[keep].drop_duplicates("stock_code").rename(
        columns={"sector_regime": "active_sector_regime"}
    )
    return snapshot.merge(sector_context, on="stock_code", how="left")


def slice_market_data(market_data: dict[str, pd.DataFrame], trade_date: str) -> dict[str, pd.DataFrame]:
    """Return in-memory market tables clipped to data available on or before the signal date."""

    signal_date = pd.Timestamp(trade_date).normalize()
    sliced: dict[str, pd.DataFrame] = {}
    for name, frame in market_data.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        current = frame.copy()
        if not current.empty and "trade_date" in current.columns:
            current["trade_date"] = pd.to_datetime(current["trade_date"]).dt.normalize()
            current = current[current["trade_date"] <= signal_date].copy()
        sliced[name] = current
    return sliced


COMPACT_COLUMNS = [
    ("stock_code", "代码"),
    ("stock_name", "名称"),
    ("active_sector_name", "板块"),
    ("top_concepts", "概念"),
    ("total_score", "评分"),
    ("rating", "评级"),
    ("rps_20", "RPS20"),
    ("sector_rps_composite", "板块RPS"),
    ("suggestion", "计划"),
]


def compact_selection(selected: pd.DataFrame) -> pd.DataFrame:
    """Return a readable subset of the selection for terminal display."""

    view = pd.DataFrame(index=selected.index)
    for source, label in COMPACT_COLUMNS:
        if source == "active_sector_name" and source not in selected.columns:
            source = "sector"
        if source not in selected.columns:
            view[label] = ""
            continue
        column = selected[source]
        if source in {"total_score"}:
            view[label] = pd.to_numeric(column, errors="coerce").round(1)
        elif source in {"rps_20", "sector_rps_composite"}:
            view[label] = pd.to_numeric(column, errors="coerce").round(0)
        elif label == "板块":
            view[label] = column.astype(str).str.slice(0, 12)
        elif label == "概念":
            view[label] = column.astype(str).str.slice(0, 18)
        else:
            view[label] = column
    return view


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Signal date, for example 2026-06-12")
    parser.add_argument("--full", action="store_true", help="Print every column instead of the compact view")
    args = parser.parse_args()
    result = run_selection(args.date)
    if result.empty:
        print("No candidates.")
        return

    csv_path = StorageManager(load_config()).result_path / f"{args.date}_selection.csv"
    core_count = int((result["rating"] == "A").sum()) if "rating" in result.columns else 0
    print(f"\n候选 {len(result)} 只 (A 级核心 {core_count} 只) — 完整字段见 {csv_path}\n")
    if args.full:
        print(result.to_string(index=False))
        return
    # east_asian_width aligns the Chinese stock and sector names in the terminal.
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print(compact_selection(result).to_string(index=False))


if __name__ == "__main__":
    main()
