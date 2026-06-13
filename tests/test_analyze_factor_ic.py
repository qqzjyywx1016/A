from scripts.analyze_factor_ic import build_arg_parser


def test_analyze_factor_ic_accepts_space_separated_horizons():
    parser = build_arg_parser()

    args = parser.parse_args(
        ["--start", "2024-01-01", "--end", "2024-06-30", "--horizons", "1", "3", "5", "10"]
    )

    assert args.horizons == [1, 3, 5, 10]
