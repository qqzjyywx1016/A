import pandas as pd

from scripts.analyze_factor_ic import FACTOR_SCORE_COLUMNS, _ic_summary, build_arg_parser


def test_analyze_factor_ic_accepts_space_separated_horizons():
    parser = build_arg_parser()

    args = parser.parse_args(
        ["--start", "2024-01-01", "--end", "2024-06-30", "--horizons", "1", "3", "5", "10"]
    )

    assert args.horizons == [1, 3, 5, 10]


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
