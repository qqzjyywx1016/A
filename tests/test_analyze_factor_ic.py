import pandas as pd

from scripts.analyze_factor_ic import (
    FACTOR_SCORE_COLUMNS,
    _df_to_markdown,
    _ic_summary,
    _ic_summary_by,
    _research_momentum,
    _write_markdown,
    build_arg_parser,
)


def test_analyze_factor_ic_accepts_space_separated_horizons():
    parser = build_arg_parser()

    args = parser.parse_args(
        ["--start", "2024-01-01", "--end", "2024-06-30", "--horizons", "1", "3", "5", "10"]
    )

    assert args.horizons == [1, 3, 5, 10]


def test_default_horizons_cover_long_holding_periods():
    args = build_arg_parser().parse_args(["--start", "2022-01-01", "--end", "2026-06-12"])

    assert args.horizons == [1, 3, 5, 10, 20, 60]
    assert args.reuse_scored is False


def test_research_momentum_skips_recent_days_and_uses_full_lookback():
    # One stock, ascending close. lookback=3, skip=1 at the last row uses
    # close[t-1]/close[t-4]-1, i.e. it ignores the most recent day's move.
    bars = pd.DataFrame(
        {
            "stock_code": ["600001.SH"] * 6,
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
            ),
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 100.0],
        }
    )

    out = _research_momentum(bars, [("mom_3_skip1", 3, 1)])
    last = out.iloc[-1]

    # skip the 14->100 jump: close[t-1]=14 over close[t-4]=11 -> 14/11 - 1.
    assert last["mom_3_skip1"] == 14.0 / 11.0 - 1
    # Warmup rows without skip+lookback history are NaN (dropped by the IC step).
    assert out["mom_3_skip1"].isna().sum() == 4


def _ic_panel(factor_values: list[float], forward_values: list[float]) -> pd.DataFrame:
    rows = []
    for day in ["2024-01-02", "2024-01-03"]:
        for i, (factor, fwd) in enumerate(zip(factor_values, forward_values, strict=True)):
            row = {"stock_code": f"{i:06d}.SZ", "trade_date": day, "forward_return_1d": fwd}
            for column in FACTOR_SCORE_COLUMNS:
                row[column] = 50.0  # constant -> NaN IC, dropped
            row["momentum_score"] = factor
            rows.append(row)
    return pd.DataFrame(rows)


def test_ic_summary_rank_ic_is_scipy_free_and_correct():
    # momentum_score perfectly rank-aligned with forward return -> rank IC == 1.
    panel = _ic_panel([1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05])

    summary = _ic_summary(panel, [1])
    momentum = summary[(summary["factor"] == "momentum_score") & (summary["horizon"] == 1)].iloc[0]

    assert momentum["ic_mean"] == 1.0
    assert momentum["observations"] == 2
    # A constant factor has no cross-sectional variance, so its IC is undefined.
    volume = summary[summary["factor"] == "volume_score"].iloc[0]
    assert volume["observations"] == 0


def test_ic_summary_detects_negative_rank_ic():
    panel = _ic_panel([1, 2, 3, 4, 5], [0.05, 0.04, 0.03, 0.02, 0.01])

    summary = _ic_summary(panel, [1])
    momentum = summary[summary["factor"] == "momentum_score"].iloc[0]

    assert momentum["ic_mean"] == -1.0


def test_ic_summary_by_segments_splits_signal_by_bucket():
    # momentum positive in 2023 (trending), negative in 2024 (reversal) -> regime-dependent.
    rows = []
    for day, fwd in [("2023-01-03", [0.01, 0.02, 0.03, 0.04, 0.05]), ("2024-01-03", [0.05, 0.04, 0.03, 0.02, 0.01])]:
        for i, (factor, ret) in enumerate(zip([1, 2, 3, 4, 5], fwd, strict=True)):
            row = {"stock_code": f"{i:06d}.SZ", "trade_date": day, "forward_return_1d": ret}
            for column in FACTOR_SCORE_COLUMNS:
                row[column] = 50.0
            row["momentum_score"] = factor
            rows.append(row)
    panel = pd.DataFrame(rows)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["year"] = panel["trade_date"].dt.year.astype(str)

    by_year = _ic_summary_by(panel, [1], "year")
    m2023 = by_year[(by_year["segment"] == "2023") & (by_year["factor"] == "momentum_score")].iloc[0]
    m2024 = by_year[(by_year["segment"] == "2024") & (by_year["factor"] == "momentum_score")].iloc[0]

    assert m2023["ic_mean"] == 1.0
    assert m2024["ic_mean"] == -1.0
    assert set(by_year["segment"]) == {"2023", "2024"}


def test_df_to_markdown_is_tabulate_free_pipe_table():
    df = pd.DataFrame([{"factor": "momentum_score", "ic_mean": 0.0123, "observations": 2, "ic_ir": pd.NA}])

    table = _df_to_markdown(df)
    lines = table.splitlines()

    assert lines[0] == "| factor | ic_mean | observations | ic_ir |"
    assert lines[1] == "| --- | --- | --- | --- |"
    assert lines[2] == "| momentum_score | 0.0123 | 2 |  |"  # NA renders blank


def test_write_markdown_renders_without_tabulate(tmp_path):
    summary = _ic_summary(_ic_panel([1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05]), [1])
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["momentum_score", "volume_score"], columns=["momentum_score", "volume_score"])
    path = tmp_path / "factor_ic_report.md"

    _write_markdown(path, summary, corr, "2025-06-01", "2026-06-12", [1])
    text = path.read_text(encoding="utf-8")

    assert "# Factor IC Report" in text
    assert "| factor | momentum_score | volume_score |" in text  # corr corner labeled
    assert "momentum_score" in text
