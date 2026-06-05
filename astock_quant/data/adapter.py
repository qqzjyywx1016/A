"""Abstract market data adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataAdapter(ABC):
    """Interface used by strategy code to access market data."""

    @abstractmethod
    def get_stock_basic(self) -> pd.DataFrame:
        """Return stock metadata such as code, name, sector, listing date and market cap."""

    @abstractmethod
    def get_daily_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return A-share daily OHLCV bars for the inclusive date range."""

    @abstractmethod
    def get_index_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return benchmark index daily bars for the inclusive date range."""

    @abstractmethod
    def get_sector_map(self) -> pd.DataFrame:
        """Return stock-to-sector mapping."""

    @abstractmethod
    def get_sector_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return sector-level daily bars or returns."""

    @abstractmethod
    def get_fund_flow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return fund-flow data for the inclusive date range."""

    @abstractmethod
    def get_trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return trading calendar rows for the inclusive date range."""

    @abstractmethod
    def get_limit_status(self, date: str) -> pd.DataFrame:
        """Return limit-up, limit-down and suspension status for one trade date."""
