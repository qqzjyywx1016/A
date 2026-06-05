"""Local storage manager for standardized data artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from astock_quant.config.loader import resolve_path


class StorageManager:
    """Read and write parquet, selection CSV and backtest trade CSV files."""

    SELECTION_COLUMNS = [
        "stock_code",
        "stock_name",
        "sector",
        "total_score",
        "momentum_score",
        "volume_score",
        "sector_score",
        "fund_score",
        "pattern_score",
        "sentiment_score",
        "rps_5",
        "rps_10",
        "rps_20",
        "rps_60",
        "rps_composite",
        "rps_pattern",
        "active_sector_name",
        "active_sector_type",
        "sector_rps_3",
        "sector_rps_5",
        "sector_rps_10",
        "sector_rps_20",
        "sector_rps_composite",
        "sector_rps_pattern",
        "rating",
        "suggestion",
    ]

    def __init__(self, config: dict[str, Any]):
        self.config = config
        data_config = config.get("data", config)
        self.raw_path = resolve_path(config, data_config.get("raw_path", "data/raw"))
        self.processed_path = resolve_path(config, data_config.get("processed_path", "data/processed"))
        self.result_path = resolve_path(config, data_config.get("result_path", "data/results"))
        self.report_path = resolve_path(config, data_config.get("report_path", "reports"))
        for path in [self.raw_path, self.processed_path, self.result_path, self.report_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _resolve_data_file(self, relative_path: str | Path, base: str = "processed") -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        base_path = {
            "raw": self.raw_path,
            "processed": self.processed_path,
            "results": self.result_path,
            "reports": self.report_path,
        }.get(base, self.processed_path)
        return base_path / path

    def save_parquet(self, df: pd.DataFrame, relative_path: str | Path, base: str = "processed") -> Path:
        """Save a DataFrame as parquet under a configured data directory."""

        path = self._resolve_data_file(relative_path, base)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def read_parquet(self, relative_path: str | Path, base: str = "processed") -> pd.DataFrame:
        """Read a parquet file from a configured data directory."""

        return pd.read_parquet(self._resolve_data_file(relative_path, base))

    def save_daily_selection(self, df: pd.DataFrame, trade_date: str) -> Path:
        """Save daily selection results as CSV."""

        path = self.result_path / f"{trade_date}_selection.csv"
        output = df.copy()
        if output.empty and len(output.columns) == 0:
            output = pd.DataFrame(columns=self.SELECTION_COLUMNS)
        output.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_rejected_candidates(self, df: pd.DataFrame, trade_date: str) -> Path:
        """Save pre-selection rejected candidates as CSV."""

        path = self.result_path / f"{trade_date}_rejected.csv"
        output = df.copy()
        if output.empty and len(output.columns) == 0:
            output = pd.DataFrame(columns=["stock_code", "reject_reason"])
        output.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_backtest_trades(self, df: pd.DataFrame, name: str = "backtest_trades") -> Path:
        """Save backtest trade records as CSV."""

        path = self.result_path / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_backtest_equity(self, df: pd.DataFrame, name: str = "backtest_equity") -> Path:
        """Save backtest daily equity curve as CSV."""

        path = self.result_path / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
