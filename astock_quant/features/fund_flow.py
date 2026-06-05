"""Fund-flow factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.features.momentum import _rank_score
from astock_quant.utils.calendar import ensure_no_future_data


class FundFlowFactor:
    """Score main-money and large-order inflow without failing when data is missing."""

    def calculate(
        self,
        fund_flow: pd.DataFrame | None,
        *,
        trade_date: str,
        stock_codes: list[str] | pd.Series | None = None,
    ) -> pd.DataFrame:
        """Return fund-flow factor rows, using neutral 50 scores when unavailable."""

        if fund_flow is not None:
            ensure_no_future_data(fund_flow, trade_date)
        if fund_flow is None or fund_flow.empty:
            if stock_codes is None:
                return pd.DataFrame(columns=["stock_code", "trade_date", "score"])
            return pd.DataFrame(
                {
                    "stock_code": list(stock_codes),
                    "trade_date": pd.Timestamp(trade_date).normalize(),
                    "score": 50.0,
                    "main_net_inflow": 0.0,
                    "super_large_net_inflow": 0.0,
                    "large_net_inflow": 0.0,
                    "main_net_inflow_ratio": 0.0,
                    "consecutive_inflow_days": 0,
                }
            )

        data = fund_flow.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
        signal_date = pd.Timestamp(trade_date).normalize()
        for column in ["main_net_inflow", "super_large_net_inflow", "large_net_inflow", "turnover_amount"]:
            if column not in data.columns:
                data[column] = 0.0
        data = data.sort_values(["stock_code", "trade_date"])
        data["main_net_inflow_ratio"] = data["main_net_inflow"] / data["turnover_amount"].replace(0, np.nan)

        def consecutive_positive(values: pd.Series) -> pd.Series:
            count = 0
            output: list[int] = []
            for value in values:
                count = count + 1 if value > 0 else 0
                output.append(count)
            return pd.Series(output, index=values.index)

        data["consecutive_inflow_days"] = data.groupby("stock_code")["main_net_inflow"].transform(consecutive_positive)
        snapshot = data[data["trade_date"] == signal_date].copy()
        if snapshot.empty:
            return pd.DataFrame(columns=["stock_code", "trade_date", "score"])

        flow_score = _rank_score(snapshot["main_net_inflow"]) * 0.35
        ratio_score = _rank_score(snapshot["main_net_inflow_ratio"].fillna(0)) * 0.30
        consecutive_score = (snapshot["consecutive_inflow_days"].clip(0, 5) / 5) * 100 * 0.20
        large_score = _rank_score(snapshot["super_large_net_inflow"] + snapshot["large_net_inflow"]) * 0.15
        snapshot["score"] = (flow_score + ratio_score + consecutive_score + large_score).clip(0, 100).round(2)
        columns = [
            "stock_code",
            "trade_date",
            "score",
            "main_net_inflow",
            "super_large_net_inflow",
            "large_net_inflow",
            "main_net_inflow_ratio",
            "consecutive_inflow_days",
        ]
        return snapshot[columns].reset_index(drop=True)
