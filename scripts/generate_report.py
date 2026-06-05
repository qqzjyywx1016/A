#!/usr/bin/env python3
"""Generate Markdown daily report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astock_quant.config.loader import load_config
from astock_quant.data.storage import StorageManager
from astock_quant.report.daily_report import DailyReportGenerator


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    config = load_config()
    storage = StorageManager(config)
    selection_path = storage.result_path / f"{args.date}_selection.csv"
    candidates = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
    path = DailyReportGenerator(config).generate(trade_date=args.date, candidates=candidates)
    print(path)


if __name__ == "__main__":
    main()
