#!/usr/bin/env python3
"""Check and update local market data sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astock_quant.config.loader import load_config
from astock_quant.data.astock_data_adapter import AStockDataAdapter
from astock_quant.utils.logger import get_logger


def main() -> None:
    """Entry point for data updates."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Start date, for example 2026-01-01")
    parser.add_argument("--end", help="End date, for example 2026-06-04")
    args = parser.parse_args()

    logger = get_logger(__name__)
    config = load_config()
    if args.start:
        config.setdefault("external", {})["update_start_date"] = args.start
    if args.end:
        config.setdefault("external", {})["update_end_date"] = args.end
    adapter = AStockDataAdapter(config)
    adapter.update_data()


if __name__ == "__main__":
    main()
