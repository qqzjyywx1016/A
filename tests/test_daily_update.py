import pandas as pd

from astock_quant.data.storage import StorageManager
from scripts.daily_update import build_arg_parser, compute_fetch_window, existing_data_range


def _storage(tmp_path):
    config = {
        "data": {
            "processed_path": str(tmp_path),
            "raw_path": str(tmp_path),
            "result_path": str(tmp_path),
            "report_path": str(tmp_path),
        }
    }
    return StorageManager(config)


def test_existing_data_range_reads_min_max_and_codes(tmp_path):
    storage = _storage(tmp_path)
    storage.save_parquet(
        pd.DataFrame(
            [
                {"stock_code": "600000.SH", "trade_date": "2026-06-10"},
                {"stock_code": "600000.SH", "trade_date": "2026-06-12"},
                {"stock_code": "000001.SZ", "trade_date": "2026-06-11"},
            ]
        ),
        "daily_bars.parquet",
    )

    data_min, data_max, codes = existing_data_range(storage)

    assert data_min == pd.Timestamp("2026-06-10")
    assert data_max == pd.Timestamp("2026-06-12")
    assert codes == ["000001.SZ", "600000.SH"]


def test_existing_data_range_handles_missing_file(tmp_path):
    data_min, data_max, codes = existing_data_range(_storage(tmp_path))

    assert data_min is None and data_max is None and codes == []


def test_compute_fetch_window_incremental_uses_lookback_from_data_max():
    start, end = compute_fetch_window(
        pd.Timestamp("2022-01-04"), pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-18"), lookback_days=90, full=False
    )

    assert end == "2026-06-18"
    # 90 days before the last data date keeps the factor lookback consistent under qfq.
    assert start == "2026-03-14"


def test_compute_fetch_window_full_refetches_entire_history():
    start, end = compute_fetch_window(
        pd.Timestamp("2022-01-04"), pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-18"), lookback_days=90, full=True
    )

    assert start == "2022-01-04"
    assert end == "2026-06-18"


def test_daily_update_arg_parser_defaults():
    args = build_arg_parser().parse_args([])

    assert args.lookback_days == 90
    assert args.full is False
    assert args.no_run is False
    assert args.relogin_every == 500
