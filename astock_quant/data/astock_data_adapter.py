"""Adapter for the external simonlin1212/a-stock-data repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from astock_quant.config.loader import resolve_path
from astock_quant.data.adapter import MarketDataAdapter
from astock_quant.data.astock_skill_sources import AStockSkillSource
from astock_quant.data.storage import StorageManager
from astock_quant.utils.logger import get_logger


class AStockDataAdapter(MarketDataAdapter):
    """Read standardized local data while isolating external data-source details."""

    def __init__(
        self,
        config: dict[str, Any],
        storage: StorageManager | None = None,
        allow_empty: bool = True,
        source: AStockSkillSource | None = None,
    ):
        self.config = config
        self.storage = storage or StorageManager(config)
        external_config = config.get("external", {})
        self.astock_data_path = resolve_path(config, external_config.get("astock_data_path", "external/a-stock-data"))
        self.allow_empty = allow_empty
        self.live_enabled = bool(external_config.get("live_enabled", True))
        self.logger = get_logger(__name__)
        self.source = source or AStockSkillSource(config, logger=self.logger)

    def _read_standard_table(self, name: str) -> pd.DataFrame:
        candidates = [
            self.storage.processed_path / f"{name}.parquet",
            self.storage.processed_path / f"{name}.csv",
            self.storage.raw_path / f"{name}.parquet",
            self.storage.raw_path / f"{name}.csv",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                return pd.read_csv(path)
        if self.allow_empty:
            return pd.DataFrame()
        raise NotImplementedError(
            f"{name} is not available. Add standardized data under data/processed or implement "
            "the external a-stock-data reader in AStockDataAdapter."
        )

    @staticmethod
    def _filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df.empty or "trade_date" not in df.columns:
            return df
        result = df.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        return result[(result["trade_date"] >= start) & (result["trade_date"] <= end)].reset_index(drop=True)

    def get_stock_basic(self) -> pd.DataFrame:
        """Return stock metadata from standardized local files or live a-stock-data endpoints."""

        local = self._read_standard_table("stock_basic")
        if not local.empty or not self.live_enabled:
            return local
        return self._safe_live_fetch("stock_basic", self.source.fetch_stock_basic)

    def get_daily_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return daily bars from standardized local files or Baidu K-line fallback."""

        local = self._filter_dates(self._read_standard_table("daily_bars"), start_date, end_date)
        if not local.empty or not self.live_enabled:
            self._warn_if_suspicious_unadjusted(local, "daily_bars")
            return local
        return self._safe_live_fetch("daily_bars", lambda: self.source.fetch_daily_bars(start_date, end_date))

    def get_index_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return index bars from standardized local files or live index K-line fallback."""

        local = self._filter_dates(self._read_standard_table("index_bars"), start_date, end_date)
        if not local.empty or not self.live_enabled:
            return local
        return self._safe_live_fetch("index_bars", lambda: self.source.fetch_index_bars(start_date, end_date))

    def get_sector_map(self) -> pd.DataFrame:
        """Return stock-to-sector map from standardized local files or Eastmoney slist."""

        local = self._read_standard_table("sector_map")
        if not local.empty or not self.live_enabled:
            return local
        return self._safe_live_fetch("sector_map", self.source.fetch_sector_map)

    def get_sector_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return sector daily data from standardized local files or Eastmoney industry ranking."""

        local = self._filter_dates(self._read_standard_table("sector_daily"), start_date, end_date)
        if not local.empty or not self.live_enabled:
            return local
        return self._safe_live_fetch("sector_daily", lambda: self.source.fetch_sector_daily(start_date, end_date))

    def get_fund_flow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return fund-flow data from standardized local files or Eastmoney push2his."""

        local = self._filter_dates(self._read_standard_table("fund_flow"), start_date, end_date)
        if not local.empty or not self.live_enabled:
            return local
        return self._safe_live_fetch("fund_flow", lambda: self.source.fetch_fund_flow(start_date, end_date))

    def get_trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return trading calendar, or derive it from daily bars if no file exists."""

        calendar = self._read_standard_table("trading_calendar")
        if not calendar.empty:
            return self._filter_dates(calendar, start_date, end_date)
        if self.live_enabled:
            live = self._safe_live_fetch(
                "trading_calendar", lambda: self.source.fetch_trading_calendar(start_date, end_date)
            )
            if not live.empty:
                return live
        bars = self.get_daily_bars(start_date, end_date)
        if bars.empty or "trade_date" not in bars.columns:
            return pd.DataFrame(columns=["trade_date", "is_open"])
        dates = sorted(pd.to_datetime(bars["trade_date"]).dt.normalize().unique())
        return pd.DataFrame({"trade_date": dates, "is_open": True})

    def get_limit_status(self, date: str) -> pd.DataFrame:
        """Return daily limit status from dedicated files or daily bars fallback columns."""

        dated_name = f"limit_status_{date}"
        table = self._read_standard_table(dated_name)
        if not table.empty:
            return table
        if self.live_enabled:
            live = self._safe_live_fetch(dated_name, lambda: self.source.fetch_limit_status(date))
            if not live.empty:
                return live
        bars = self.get_daily_bars(date, date)
        columns = ["stock_code", "trade_date", "is_limit_up", "is_limit_down", "is_suspended"]
        if bars.empty:
            return pd.DataFrame(columns=columns)
        available = [column for column in columns if column in bars.columns]
        return bars[available].copy()

    def update_data(self) -> None:
        """Fetch live data through a-stock-data endpoints and persist standardized parquet files."""

        if not Path(self.astock_data_path).exists():
            self.logger.warning(
                "external/a-stock-data is missing. Clone https://github.com/simonlin1212/a-stock-data "
                "into that path before wiring the external reader."
            )
        external_config = self.config.get("external", {})
        end_date = external_config.get("update_end_date") or pd.Timestamp.today().date().isoformat()
        start_date = external_config.get("update_start_date") or (pd.Timestamp(end_date) - pd.Timedelta(days=180)).date().isoformat()
        tasks = [
            ("stock_basic", "stock_basic.parquet", self.source.fetch_stock_basic),
            ("daily_bars", "daily_bars.parquet", lambda: self.source.fetch_daily_bars(start_date, end_date)),
            ("index_bars", "index_bars.parquet", lambda: self.source.fetch_index_bars(start_date, end_date)),
            ("sector_map", "sector_map.parquet", self.source.fetch_sector_map),
            ("sector_daily", "sector_daily.parquet", lambda: self.source.fetch_sector_daily(start_date, end_date)),
            ("fund_flow", "fund_flow.parquet", lambda: self.source.fetch_fund_flow(start_date, end_date)),
            ("trading_calendar", "trading_calendar.parquet", lambda: self.source.fetch_trading_calendar(start_date, end_date)),
            (f"limit_status_{end_date}", f"limit_status_{end_date}.parquet", lambda: self.source.fetch_limit_status(end_date)),
        ]
        for name, path, fetcher in tasks:
            try:
                data = fetcher()
            except Exception as exc:
                self.logger.warning("update_data skipped %s after live source failure: %s", name, exc)
                continue
            if data.empty:
                self.logger.warning("update_data got empty %s; keeping existing local files", name)
                continue
            self.storage.save_parquet(data, path)
            self.logger.info("updated %s rows=%s", name, len(data))

    def _safe_live_fetch(self, name: str, fetcher) -> pd.DataFrame:
        try:
            return fetcher()
        except Exception as exc:
            self.logger.warning("live fetch failed for %s: %s", name, exc)
            return pd.DataFrame()

    def _warn_if_suspicious_unadjusted(self, df: pd.DataFrame, name: str) -> None:
        if df.empty or not {"stock_code", "trade_date", "close"}.issubset(df.columns):
            return
        data = df.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
        data = data.sort_values(["stock_code", "trade_date"])
        returns = data.groupby("stock_code")["close"].pct_change().abs()
        valid = returns.dropna()
        if len(valid) < 10:
            return
        abnormal_ratio = float((valid > 0.5).mean())
        if abnormal_ratio > 0.01:
            self.logger.warning(
                "%s may not be qfq-adjusted: abnormal adjacent close jump ratio %.2f%%",
                name,
                abnormal_ratio * 100,
            )
