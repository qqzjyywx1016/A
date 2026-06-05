"""Tradable stock universe filters."""

from __future__ import annotations

from typing import Any

import pandas as pd

from astock_quant.utils.calendar import ensure_no_future_data


class UniverseFilter:
    """Apply configurable A-share universe exclusions before factor scoring."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def apply(self, stocks: pd.DataFrame, as_of_date: str | None = None) -> pd.DataFrame:
        """Return stocks that pass ST, suspension, liquidity and market-cap filters."""

        ensure_no_future_data(stocks, as_of_date)
        if stocks.empty:
            return stocks.copy()

        result = stocks.copy()
        mask = pd.Series(True, index=result.index)

        if self.config.get("exclude_st", False):
            if "is_st" in result.columns:
                mask &= ~result["is_st"].fillna(False).astype(bool)
            if "stock_name" in result.columns:
                mask &= ~result["stock_name"].astype(str).str.upper().str.contains("ST", regex=False)

        if self.config.get("exclude_suspended", False) and "is_suspended" in result.columns:
            mask &= ~result["is_suspended"].fillna(False).astype(bool)

        if self.config.get("exclude_bj", False):
            if "exchange" in result.columns:
                mask &= result["exchange"].astype(str).str.upper() != "BJ"
            elif "stock_code" in result.columns:
                codes = result["stock_code"].astype(str).str.upper()
                mask &= ~codes.str.endswith(".BJ")
                mask &= ~codes.str.match(r"^(4|8|83|87)")

        numeric_filters = [
            ("listing_days", "min_listing_days", ">="),
            ("turnover_amount", "min_turnover_amount", ">="),
            ("avg_turnover_amount_20d", "min_avg_turnover_amount_20d", ">="),
            ("avg_turnover_rate_20d", "min_avg_turnover_rate_20d", ">="),
            ("float_market_cap", "min_float_market_cap", ">="),
            ("float_market_cap", "max_float_market_cap", "<="),
        ]
        for column, config_key, operator in numeric_filters:
            if column not in result.columns or config_key not in self.config:
                continue
            values = pd.to_numeric(result[column], errors="coerce")
            threshold = self.config[config_key]
            if threshold is None:
                continue
            if operator == ">=":
                mask &= values >= threshold
            else:
                mask &= values <= threshold

        if as_of_date is not None:
            as_of = pd.Timestamp(as_of_date).normalize()
            if "next_report_date" in result.columns:
                blackout_days = int(self.config.get("earnings_blackout_days", 0) or 0)
                if blackout_days > 0:
                    report_dates = pd.to_datetime(result["next_report_date"], errors="coerce").dt.normalize()
                    mask &= ~((report_dates - as_of).abs().dt.days <= blackout_days).fillna(False)
            if "lockup_date" in result.columns:
                blackout_days = int(self.config.get("lockup_blackout_days", 0) or 0)
                if blackout_days > 0:
                    lockup_dates = pd.to_datetime(result["lockup_date"], errors="coerce").dt.normalize()
                    mask &= ~((lockup_dates - as_of).abs().dt.days <= blackout_days).fillna(False)

        for event_column in ["is_restructuring", "is_major_event"]:
            if event_column in result.columns:
                mask &= ~result[event_column].fillna(False).astype(bool)

        return result.loc[mask].reset_index(drop=True)
