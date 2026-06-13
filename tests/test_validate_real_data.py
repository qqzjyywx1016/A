import json

import pandas as pd

from scripts.validate_real_data import ValidationOptions, run_validation


def _config(tmp_path):
    return {
        "data": {
            "raw_path": str(tmp_path / "raw"),
            "processed_path": str(tmp_path / "processed"),
            "result_path": str(tmp_path / "results"),
            "report_path": str(tmp_path / "reports"),
        },
        "external": {"live_enabled": False, "astock_data_path": "external/a-stock-data"},
    }


def test_validate_real_data_empty_tables_hard_fail_but_writes_outputs(tmp_path):
    config = _config(tmp_path)

    result = run_validation(config, "2026-06-01", "2026-06-04", ValidationOptions())

    assert result.exit_code == 1
    assert result.summary["hard_failed"] is True
    assert (tmp_path / "results" / "data_validation_summary.json").exists()
    report = (tmp_path / "reports" / "data_validation_report.md").read_text(encoding="utf-8")
    assert "数据偏差/局限" in report.splitlines()[0]


def test_validate_real_data_requires_confirmed_qfq_adjustment(tmp_path):
    config = _config(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame([{"stock_code": "600001.SH", "stock_name": "样本"}]).to_parquet(processed / "stock_basic.parquet")
    pd.DataFrame([{"index_code": "000852.SH", "trade_date": "2026-06-02", "close": 100}]).to_parquet(
        processed / "index_bars.parquet"
    )
    pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "trade_date": "2026-06-02",
                "open": 10,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "prev_close": 10,
                "volume": 1000,
                "amount": 10_000,
                "pct_chg": 0.02,
            }
        ]
    ).to_parquet(processed / "daily_bars.parquet")

    result = run_validation(config, "2026-06-01", "2026-06-04", ValidationOptions())

    assert result.exit_code == 1
    assert any("未确认复权口径" in issue for issue in result.summary["hard_failures"])


def test_validate_real_data_accepts_synthetic_qfq_data(tmp_path):
    config = _config(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    dates = pd.bdate_range("2026-06-01", periods=3)
    pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "stock_name": "样本",
                "is_st_source": "as_of",
                "list_date_source": "as_of",
                "float_market_cap_source": "daily_point_in_time",
                "float_market_cap_is_point_in_time": True,
            }
        ]
    ).to_parquet(processed / "stock_basic.parquet")
    pd.DataFrame([{"index_code": "000852.SH", "trade_date": date, "close": 100 + i} for i, date in enumerate(dates)]).to_parquet(
        processed / "index_bars.parquet"
    )
    rows = []
    prev = 10.0
    for index, trade_date in enumerate(dates):
        close = prev * 1.01
        rows.append(
            {
                "stock_code": "600001.SH",
                "trade_date": trade_date,
                "open": prev,
                "high": close + 0.2,
                "low": prev - 0.2,
                "close": close,
                "prev_close": prev,
                "volume": 1000,
                "amount": 10_000,
                "pct_chg": close / prev - 1,
                "adjust_type": "qfq",
                "float_market_cap": 5_000_000_000,
                "is_st": False,
            }
        )
        prev = close
    pd.DataFrame(rows).to_parquet(processed / "daily_bars.parquet")

    result = run_validation(config, "2026-06-01", "2026-06-04", ValidationOptions())

    assert result.exit_code == 0
    summary = json.loads((tmp_path / "results" / "data_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["hard_failed"] is False
    assert summary["qfq_confirmed"] is True
