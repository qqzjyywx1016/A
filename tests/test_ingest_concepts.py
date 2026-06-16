import pandas as pd

from scripts.ingest_concepts import build_concept_row, _is_concept, build_arg_parser


def test_build_concept_row_filters_non_concepts_and_sorts_by_heat():
    blocks = [
        {"name": "融资融券", "code": "BK0001", "change_pct": "9.9"},
        {"name": "人形机器人", "code": "BK0002", "change_pct": "5.1"},
        {"name": "减速器", "code": "BK0003", "change_pct": "8.4"},
        {"name": "深股通", "code": "BK0004", "change_pct": "1.0"},
        {"name": "AI算力", "code": "BK0005", "change_pct": "2.0"},
    ]

    row = build_concept_row("300400.SZ", blocks, top_n=2)

    # Trading-status boards (融资融券/深股通) are dropped; concepts sort hottest-first.
    assert row["top_concepts"] == "减速器,人形机器人"
    assert row["top_concept"] == "减速器"
    assert row["concept_count"] == 3
    assert row["concept_tags"] == "减速器,人形机器人,AI算力"
    assert row["stock_code"] == "300400.SZ"
    assert row["fetch_date"]


def test_build_concept_row_handles_empty_blocks():
    row = build_concept_row("600000.SH", [], top_n=3)

    assert row["concept_count"] == 0
    assert row["top_concepts"] == ""
    assert row["top_concept"] == ""


def test_is_concept_rejects_trading_status_and_region_labels():
    assert _is_concept("固态电池")
    assert not _is_concept("融资融券")
    assert not _is_concept("深圳板块")
    assert not _is_concept("")


def test_concept_arg_parser_defaults():
    args = build_arg_parser().parse_args([])

    assert args.top_n == 3
    assert args.save_every == 100
    assert args.sleep == 0.0
