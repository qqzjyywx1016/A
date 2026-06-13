import pandas as pd

from astock_quant.strategy.overheat_filter import OverheatFilter
import scripts.run_selection as run_selection_module


class _FactorStub:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def calculate(self, first_frame: pd.DataFrame, *, trade_date: str, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def _overheat_config() -> dict:
    return {
        "enabled": True,
        "rps5_threshold": 95,
        "large_gain_pct": 7,
        "volume_ratio_threshold": 2.5,
        "climax_rps5_threshold": 97,
        "climax_rps10_threshold": 95,
        "climax_gain_pct": 8,
        "stagnation_volume_ratio": 3,
        "stagnation_gain_pct": 3,
        "sector_stock_count_threshold": 10,
        "sector_rps5_threshold": 95,
        "sector_limit_up_ratio_threshold": 0.08,
    }


def test_overheat_filter_rejects_extreme_rps_gain_volume():
    frame = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "rps_5": 96,
                "rps_10": 90,
                "return_1d": 0.075,
                "volume_ratio_20d": 2.6,
                "long_upper_shadow": False,
                "turnover_amount": 100,
                "amount_60d_max": 120,
            }
        ]
    )

    passed, rejected = OverheatFilter(_overheat_config()).apply(frame)

    assert passed.empty
    assert rejected[["stock_code", "reject_reason"]].to_dict("records") == [
        {"stock_code": "600001.SH", "reject_reason": "overheat_rps5_large_gain_volume"}
    ]


def test_overheat_filter_rejects_sector_climax():
    rows = []
    for index in range(10):
        rows.append(
            {
                "stock_code": f"600{index:03d}.SH",
                "active_sector_code": "BK001",
                "rps_5": 80,
                "rps_10": 80,
                "return_1d": 0.01,
                "volume_ratio_20d": 1.0,
                "long_upper_shadow": False,
                "turnover_amount": 100,
                "amount_60d_max": 200,
                "sector_rps_5": 96,
                "is_limit_up": index == 0,
            }
        )
    frame = pd.DataFrame(rows)

    passed, rejected = OverheatFilter(_overheat_config()).apply(frame)

    assert passed.empty
    assert set(rejected["reject_reason"]) == {"sector_climax"}


def test_run_selection_filters_overheated_candidate_and_saves_rejected(monkeypatch, tmp_path):
    trade_date = "2026-06-04"
    hot = "600001.SH"
    cool = "600002.SH"
    market_data = {
        "stock_basic": pd.DataFrame(
            [
                {"stock_code": hot, "stock_name": "过热股份", "sector": "机器人", "is_st": False, "exchange": "SH", "float_market_cap": 5_000_000_000},
                {"stock_code": cool, "stock_name": "健康股份", "sector": "机器人", "is_st": False, "exchange": "SH", "float_market_cap": 5_000_000_000},
            ]
        ),
        "daily_bars": pd.DataFrame(
            [
                {"stock_code": hot, "trade_date": trade_date, "open": 10, "high": 11, "low": 9.8, "close": 10.8, "turnover_amount": 300_000_000, "avg_turnover_amount_20d": 200_000_000, "is_suspended": False, "is_limit_up": False, "is_limit_down": False},
                {"stock_code": cool, "trade_date": trade_date, "open": 10, "high": 10.5, "low": 9.8, "close": 10.3, "turnover_amount": 300_000_000, "avg_turnover_amount_20d": 200_000_000, "is_suspended": False, "is_limit_up": False, "is_limit_down": False},
            ]
        ),
        "sector_map": pd.DataFrame(),
        "sector_daily": pd.DataFrame(),
        "fund_flow": pd.DataFrame(),
        "index_bars": pd.DataFrame(),
        "limit_status": pd.DataFrame(),
    }
    monkeypatch.setattr(run_selection_module, "AStockDataAdapter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("adapter used")))
    monkeypatch.setattr(
        run_selection_module,
        "MomentumFactor",
        lambda: _FactorStub(
            pd.DataFrame(
                [
                    {"stock_code": hot, "trade_date": trade_date, "score": 95, "return_1d": 0.08, "rps_5": 96, "rps_10": 90, "rps_20": 80, "rps_60": 72, "rps_composite": 88, "return_5d": 0.1, "return_10d": 0.12, "above_ma20": True},
                    {"stock_code": cool, "trade_date": trade_date, "score": 92, "return_1d": 0.03, "rps_5": 88, "rps_10": 86, "rps_20": 82, "rps_60": 74, "rps_composite": 86, "return_5d": 0.08, "return_10d": 0.1, "above_ma20": True},
                ]
            )
        ),
    )
    monkeypatch.setattr(
        run_selection_module,
        "VolumeFactor",
        lambda: _FactorStub(
            pd.DataFrame(
                [
                    {"stock_code": hot, "trade_date": trade_date, "score": 90, "volume_ratio_20d": 2.6, "turnover_amount": 300_000_000, "amount_60d_max": 350_000_000},
                    {"stock_code": cool, "trade_date": trade_date, "score": 90, "volume_ratio_20d": 1.3, "turnover_amount": 300_000_000, "amount_60d_max": 350_000_000},
                ]
            )
        ),
    )
    monkeypatch.setattr(
        run_selection_module,
        "SectorFactor",
        lambda *args, **kwargs: _FactorStub(
            pd.DataFrame(
                [
                    {"stock_code": hot, "trade_date": trade_date, "score": 90, "sector": "机器人", "sector_regime": "strong", "active_sector_code": "BK001", "sector_rps_5": 80, "sector_rps_10": 75},
                    {"stock_code": cool, "trade_date": trade_date, "score": 90, "sector": "机器人", "sector_regime": "strong", "active_sector_code": "BK001", "sector_rps_5": 80, "sector_rps_10": 75},
                ]
            )
        ),
    )
    monkeypatch.setattr(run_selection_module, "FundFlowFactor", lambda: _FactorStub(pd.DataFrame([{"stock_code": hot, "trade_date": trade_date, "score": 90}, {"stock_code": cool, "trade_date": trade_date, "score": 90}])))
    monkeypatch.setattr(
        run_selection_module,
        "PatternFactor",
        lambda: _FactorStub(pd.DataFrame([{"stock_code": hot, "trade_date": trade_date, "score": 90, "long_upper_shadow": False, "high_volume_stagnation": False}, {"stock_code": cool, "trade_date": trade_date, "score": 90, "long_upper_shadow": False, "high_volume_stagnation": False}])),
    )
    monkeypatch.setattr(
        run_selection_module,
        "SentimentFactor",
        lambda *args, **kwargs: _FactorStub(pd.DataFrame([{"stock_code": hot, "trade_date": trade_date, "score": 90, "market_regime": "strong"}, {"stock_code": cool, "trade_date": trade_date, "score": 90, "market_regime": "strong"}])),
    )
    config = {
        "data": {"result_path": str(tmp_path), "processed_path": str(tmp_path), "raw_path": str(tmp_path), "report_path": str(tmp_path)},
        "universe": {"min_listing_days": 0, "min_turnover_amount": 0, "min_avg_turnover_amount_20d": 0, "min_float_market_cap": 0, "max_float_market_cap": 99_000_000_000},
        "score_weights": {"momentum": 0.25, "volume": 0.20, "sector": 0.20, "fund_flow": 0.15, "pattern": 0.10, "sentiment": 0.10},
        "selection": {"min_total_score": 70, "max_candidates": 20, "max_core_pool": 5},
        "rps": {"enabled": True, "filters": {"strong": {"rps_20": 65}}, "gate": {"rps_60_min": 60, "return_10d_min": -0.03}},
        "sector_rps": {"enabled": True, "filters": {"strong": {"sector_rps_5": 55, "sector_rps_10": 50}}},
        "overheat": _overheat_config(),
    }

    selected = run_selection_module.run_selection(trade_date, config=config, save=True, market_data=market_data)

    assert selected["stock_code"].tolist() == [cool]
    rejected = pd.read_csv(tmp_path / f"{trade_date}_rejected.csv")
    assert rejected[["stock_code", "reject_reason"]].to_dict("records") == [
        {"stock_code": hot, "reject_reason": "overheat_rps5_large_gain_volume"}
    ]
