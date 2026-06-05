import pandas as pd

from astock_quant.data.astock_data_adapter import AStockDataAdapter
from astock_quant.data.storage import StorageManager


class FailingSource:
    def fetch_daily_bars(self, start_date, end_date):
        raise RuntimeError("network down")


def test_adapter_falls_back_to_standard_processed_cache_when_live_source_fails(tmp_path):
    config = {
        "_project_root": str(tmp_path),
        "data": {
            "raw_path": str(tmp_path / "raw"),
            "processed_path": str(tmp_path / "processed"),
            "result_path": str(tmp_path / "results"),
            "report_path": str(tmp_path / "reports"),
        },
    }
    storage = StorageManager(config)
    cached = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "trade_date": "2026-06-04",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 1000,
                "turnover_amount": 100000,
            }
        ]
    )
    storage.save_parquet(cached, "daily_bars.parquet")

    adapter = AStockDataAdapter(config, storage=storage, source=FailingSource())
    result = adapter.get_daily_bars("2026-06-04", "2026-06-04")

    assert result["stock_code"].tolist() == ["600001.SH"]
    assert result.iloc[0]["close"] == 10.5
