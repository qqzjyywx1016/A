import pandas as pd

from astock_quant.data.storage import StorageManager


def test_save_empty_daily_selection_keeps_reportable_columns(tmp_path):
    config = {
        "_project_root": str(tmp_path),
        "data": {
            "raw_path": str(tmp_path / "raw"),
            "processed_path": str(tmp_path / "processed"),
            "result_path": str(tmp_path / "results"),
            "report_path": str(tmp_path / "reports"),
        },
    }
    path = StorageManager(config).save_daily_selection(pd.DataFrame(), "2026-06-04")

    loaded = pd.read_csv(path)

    assert "stock_code" in loaded.columns
    assert "suggestion" in loaded.columns
    assert loaded.empty
