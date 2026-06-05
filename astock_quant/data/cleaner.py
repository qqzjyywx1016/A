"""Data normalization utilities."""

from __future__ import annotations

import pandas as pd


class DataCleaner:
    """Normalize market data schemas used by the strategy modules."""

    @staticmethod
    def normalize_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dates, sort bars and coerce numeric price/volume columns."""

        result = df.copy()
        if "trade_date" in result.columns:
            result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
        for column in ["open", "high", "low", "close", "prev_close", "volume", "turnover_amount"]:
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        sort_columns = [column for column in ["stock_code", "trade_date"] if column in result.columns]
        if sort_columns:
            result = result.sort_values(sort_columns).reset_index(drop=True)
        return result
