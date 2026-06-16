#!/usr/bin/env python3
"""Ingest current Eastmoney concept-board tags per stock into a concept_map.

A-share moves on themes, not just industry, so this annotates the LIVE selection
with each stock's concept boards. baostock provides no concept data, so the tags
come from Eastmoney's slist endpoint (already wrapped by AStockSkillSource).

IMPORTANT: these are *current* tags with no point-in-time history. They are for
live display only and are never merged into the backtest path, where using
today's tags on past dates would be look-ahead bias.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astock_quant.config.loader import load_config
from astock_quant.data.astock_skill_sources import AStockSkillSource
from astock_quant.data.storage import StorageManager
from scripts.ingest_baostock import Throttle

CONCEPT_MAP_FILE = "concept_map.parquet"
CONCEPT_MAP_COLUMNS = ["stock_code", "concept_tags", "top_concepts", "top_concept", "concept_count", "fetch_date"]

# Eastmoney slist returns industry/region/trading-status blocks alongside real
# concepts; drop the obvious non-theme ones so the display shows actual themes.
NON_CONCEPT_HINTS = (
    "融资融券",
    "沪股通",
    "深股通",
    "转融券",
    "标普",
    "MSCI",
    "富时",
    "AB股",
    "AH股",
    "GDR",
    "新股",
    "次新股",
    "ST板块",
    "板块",  # generic "xx板块" region labels like 深圳板块/上海板块
)


def _is_concept(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    return not any(hint in text for hint in NON_CONCEPT_HINTS)


def build_concept_row(stock_code: str, blocks: list[dict], top_n: int) -> dict:
    """Reduce a stock's Eastmoney blocks to display-ready concept fields."""

    concepts = [block for block in blocks if _is_concept(block.get("name"))]
    # Hottest first by the board's change_pct so the leading theme shows first.
    concepts.sort(key=lambda block: _to_float(block.get("change_pct")), reverse=True)
    names = [str(block.get("name")).strip() for block in concepts if block.get("name")]
    top = names[:top_n]
    return {
        "stock_code": stock_code,
        "concept_tags": ",".join(names),
        "top_concepts": ",".join(top),
        "top_concept": top[0] if top else "",
        "concept_count": len(names),
        "fetch_date": date.today().isoformat(),
    }


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def resolve_codes(args: argparse.Namespace, storage: StorageManager, source: AStockSkillSource) -> list[str]:
    """Prefer explicit --codes, then the already-ingested universe, then the source."""

    if args.codes:
        return [code.strip() for code in args.codes.split(",") if code.strip()]
    basic_path = storage.processed_path / "stock_basic.parquet"
    if basic_path.exists():
        basic = pd.read_parquet(basic_path)
        if not basic.empty and "stock_code" in basic.columns:
            codes = basic["stock_code"].dropna().astype(str).tolist()
            return codes[: args.limit] if args.limit else codes
    codes = source._target_codes()
    return codes[: args.limit] if args.limit else codes


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage = StorageManager(config)
    source = AStockSkillSource(config)
    throttle = Throttle(
        sleep_seconds=args.sleep, batch_size=args.batch_size, batch_rest_seconds=args.batch_rest, jitter=args.jitter
    )

    codes = resolve_codes(args, storage, source)
    if not codes:
        print("no codes to fetch; run the baostock ingest first or pass --codes")
        return 1

    existing = _read_existing(storage)
    done = set(existing["stock_code"]) if not existing.empty else set()
    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    failed: list[str] = []
    flush_every = max(int(args.save_every or 0), 1)
    fetched_since_save = 0

    for index, code in enumerate(codes, start=1):
        if code in done:
            print(f"[{index}/{len(codes)}] skip {code} existing")
            continue
        try:
            blocks = source._fetch_eastmoney_concept_blocks(code)
        except Exception as exc:  # network/format errors must not kill an overnight run
            print(f"WARNING concept {code} skipped: {exc}")
            failed.append(code)
            throttle.tick()
            continue
        row = build_concept_row(code, blocks, args.top_n)
        rows.append(row)
        done.add(code)
        fetched_since_save += 1
        print(f"[{index}/{len(codes)}] {code} concepts={row['concept_count']} top={row['top_concept']}")
        throttle.tick()
        if fetched_since_save >= flush_every:
            _save(storage, rows)
            fetched_since_save = 0
            print(f"concept_map checkpoint rows={len(rows)}")

    path = _save(storage, rows)
    print(f"concept_map rows={len(rows)} -> {path}")
    if failed:
        print(f"failed {len(failed)} codes: {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")
    return 0


def _read_existing(storage: StorageManager) -> pd.DataFrame:
    path = storage.processed_path / CONCEPT_MAP_FILE
    if not path.exists():
        return pd.DataFrame(columns=CONCEPT_MAP_COLUMNS)
    existing = pd.read_parquet(path)
    for column in CONCEPT_MAP_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA
    return existing[CONCEPT_MAP_COLUMNS]


def _save(storage: StorageManager, rows: list[dict]) -> Path:
    frame = pd.DataFrame(rows, columns=CONCEPT_MAP_COLUMNS).drop_duplicates("stock_code", keep="last")
    return storage.save_parquet(frame, CONCEPT_MAP_FILE)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Eastmoney concept tags per stock (live display only)")
    parser.add_argument("--codes", default="", help="Comma-separated standard codes; default is the ingested universe")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit on number of stocks")
    parser.add_argument("--config", default=None, help="Config path")
    parser.add_argument("--top-n", type=int, default=3, help="How many top concepts to keep for display")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint concept_map after this many stocks")
    parser.add_argument("--sleep", type=float, default=0.0, help="Extra pause between stocks (Eastmoney is already paced by em_min_interval)")
    parser.add_argument("--batch-size", type=int, default=0, help="Take a longer rest after this many stocks")
    parser.add_argument("--batch-rest", type=float, default=0.0, help="Seconds to rest between batches")
    parser.add_argument("--jitter", type=float, default=0.2, help="Randomize pauses by +-this fraction")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
