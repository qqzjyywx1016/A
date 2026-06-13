import pandas as pd

from scripts.run_batch_signals import build_backtest_panel
import scripts.run_selection as run_selection_module


class _FactorStub:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def calculate(self, first_frame: pd.DataFrame, *, trade_date: str, **kwargs) -> pd.DataFrame:
        if not first_frame.empty and "trade_date" in first_frame.columns:
            assert pd.to_datetime(first_frame["trade_date"]).max() <= pd.Timestamp(trade_date)
        for value in kwargs.values():
            if isinstance(value, pd.DataFrame) and not value.empty and "trade_date" in value.columns:
                assert pd.to_datetime(value["trade_date"]).max() <= pd.Timestamp(trade_date)
        return self.frame.copy()


def test_run_selection_uses_in_memory_market_data_and_slices_future_rows(monkeypatch, tmp_path):
    trade_date = "2026-06-04"
    stock_code = "600001.SH"
    market_data = {
        "stock_basic": pd.DataFrame(
            [
                {
                    "stock_code": stock_code,
                    "stock_name": "强势股份",
                    "sector": "机器人",
                    "is_st": False,
                    "exchange": "SH",
                    "float_market_cap": 5_000_000_000,
                }
            ]
        ),
        "daily_bars": pd.DataFrame(
            [
                {
                    "stock_code": stock_code,
                    "trade_date": trade_date,
                    "open": 10,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.4,
                    "turnover_amount": 300_000_000,
                    "avg_turnover_amount_20d": 200_000_000,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                },
                {
                    "stock_code": stock_code,
                    "trade_date": "2026-06-05",
                    "open": 99,
                    "high": 99,
                    "low": 99,
                    "close": 99,
                    "turnover_amount": 999_000_000,
                    "avg_turnover_amount_20d": 999_000_000,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                },
            ]
        ),
        "sector_map": pd.DataFrame([{"stock_code": stock_code, "sector": "机器人"}]),
        "sector_daily": pd.DataFrame(
            [
                {"sector": "机器人", "trade_date": trade_date, "close": 100, "turnover_amount": 1_000_000_000},
                {"sector": "机器人", "trade_date": "2026-06-05", "close": 200, "turnover_amount": 2_000_000_000},
            ]
        ),
        "fund_flow": pd.DataFrame(
            [
                {"stock_code": stock_code, "trade_date": trade_date, "main_net_inflow": 1_000_000},
                {"stock_code": stock_code, "trade_date": "2026-06-05", "main_net_inflow": 9_000_000},
            ]
        ),
        "index_bars": pd.DataFrame(
            [
                {"index_code": "000300.SH", "trade_date": trade_date, "close": 100},
                {"index_code": "000300.SH", "trade_date": "2026-06-05", "close": 200},
            ]
        ),
        "limit_status": pd.DataFrame(),
    }
    base = {"stock_code": stock_code, "trade_date": trade_date}
    monkeypatch.setattr(run_selection_module, "AStockDataAdapter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("adapter used")))
    monkeypatch.setattr(
        run_selection_module,
        "MomentumFactor",
        lambda: _FactorStub(
            pd.DataFrame(
                [
                    {
                        **base,
                        "score": 90,
                        "rps_5": 90,
                        "rps_10": 88,
                        "rps_20": 85,
                        "rps_60": 75,
                        "rps_composite": 88,
                        "return_5d": 0.05,
                        "return_10d": 0.08,
                        "above_ma20": True,
                    }
                ]
            )
        ),
    )
    monkeypatch.setattr(run_selection_module, "VolumeFactor", lambda: _FactorStub(pd.DataFrame([{**base, "score": 90}])))
    monkeypatch.setattr(
        run_selection_module,
        "SectorFactor",
        lambda *args, **kwargs: _FactorStub(
            pd.DataFrame(
                [
                    {
                        **base,
                        "score": 90,
                        "sector": "机器人",
                        "sector_regime": "strong",
                        "sector_rps_5": 90,
                        "sector_rps_10": 88,
                        "sector_rps_composite": 89,
                    }
                ]
            )
        ),
    )
    monkeypatch.setattr(run_selection_module, "FundFlowFactor", lambda: _FactorStub(pd.DataFrame([{**base, "score": 90}])))
    monkeypatch.setattr(
        run_selection_module,
        "PatternFactor",
        lambda: _FactorStub(pd.DataFrame([{**base, "score": 90, "long_upper_shadow": False, "high_volume_stagnation": False}])),
    )
    monkeypatch.setattr(
        run_selection_module,
        "SentimentFactor",
        lambda *args, **kwargs: _FactorStub(pd.DataFrame([{**base, "score": 90, "market_regime": "strong"}])),
    )
    config = {
        "data": {"result_path": str(tmp_path), "processed_path": str(tmp_path), "raw_path": str(tmp_path), "report_path": str(tmp_path)},
        "universe": {
            "exclude_st": True,
            "exclude_suspended": True,
            "exclude_bj": True,
            "min_listing_days": 0,
            "min_turnover_amount": 0,
            "min_avg_turnover_amount_20d": 0,
            "min_float_market_cap": 0,
            "max_float_market_cap": 99_000_000_000,
        },
        "score_weights": {"momentum": 0.25, "volume": 0.20, "sector": 0.20, "fund_flow": 0.15, "pattern": 0.10, "sentiment": 0.10},
        "selection": {"min_total_score": 70, "max_candidates": 20, "max_core_pool": 5},
        "rps": {
            "enabled": True,
            "filters": {"strong": {"rps_20": 65}},
            "gate": {"rps_60_min": 60, "return_10d_min": -0.03},
        },
        "sector_rps": {
            "enabled": True,
            "filters": {"strong": {"sector_rps_5": 55, "sector_rps_10": 50}},
        },
    }

    selected = run_selection_module.run_selection(trade_date, config=config, save=False, market_data=market_data)

    assert selected["stock_code"].tolist() == [stock_code]


def test_build_backtest_panel_includes_continue_hold_columns():
    trade_date = "2026-06-04"
    bars = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "trade_date": "2026-06-03", "close": 10},
            {"stock_code": "600001.SH", "trade_date": trade_date, "close": 11},
        ]
    )
    scored = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "trade_date": trade_date,
                "rps_20": 82,
                "sector_rps_5": 74,
                "sector_rps_10": 68,
                "high_volume_bearish": False,
                "high_volume_stagnation": False,
                "long_upper_shadow": False,
                "market_regime": "strong",
            }
        ]
    )

    panel = build_backtest_panel(scored, bars, trade_date)

    assert panel.columns.tolist() == [
        "stock_code",
        "trade_date",
        "ma5",
        "ma10",
        "rps_20",
        "sector_rps_5",
        "sector_rps_10",
        "high_volume_bearish",
        "high_volume_stagnation",
        "long_upper_shadow",
        "market_regime",
    ]
    assert panel.iloc[0]["ma5"] == 10.5
