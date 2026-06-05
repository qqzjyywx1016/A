"""Trading-calendar and date validation helpers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd


def to_timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert a supported date-like value to a normalized pandas timestamp."""

    return pd.Timestamp(value).normalize()


def normalize_date_column(df: pd.DataFrame, column: str = "trade_date") -> pd.DataFrame:
    """Return a copy with a normalized date column when it exists."""

    result = df.copy()
    if column in result.columns:
        result[column] = pd.to_datetime(result[column]).dt.normalize()
    return result


def ensure_no_future_data(
    df: pd.DataFrame,
    as_of_date: str | date | datetime | pd.Timestamp | None,
    *,
    date_column: str = "trade_date",
) -> None:
    """Raise when input data contains rows after the signal date."""

    if as_of_date is None or df.empty or date_column not in df.columns:
        return
    cutoff = to_timestamp(as_of_date)
    dates = pd.to_datetime(df[date_column]).dt.normalize()
    if (dates > cutoff).any():
        max_date = dates.max().date().isoformat()
        raise ValueError(f"future data detected: max {date_column}={max_date} after {cutoff.date().isoformat()}")


def previous_trading_date(dates: Iterable[str | date | datetime | pd.Timestamp], current: pd.Timestamp) -> pd.Timestamp | None:
    """Return the previous available trading date from an iterable of dates."""

    ordered = sorted({to_timestamp(item) for item in dates})
    for index, value in enumerate(ordered):
        if value == current and index > 0:
            return ordered[index - 1]
    return None


def trading_days_between(
    dates: Iterable[str | date | datetime | pd.Timestamp],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    """Count available trading days inclusively between two dates."""

    start_ts = to_timestamp(start)
    end_ts = to_timestamp(end)
    return sum(1 for item in {to_timestamp(day) for day in dates} if start_ts <= item <= end_ts)
