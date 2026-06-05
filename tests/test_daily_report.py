import pandas as pd

from astock_quant.report.daily_report import DailyReportGenerator


def test_daily_report_generates_candidate_table_without_extra_markdown_dependency(tmp_path):
    config = {
        "_project_root": str(tmp_path),
        "data": {"report_path": str(tmp_path / "reports")},
    }
    candidates = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "stock_name": "强势股份",
                "sector": "机器人",
                "total_score": 88,
                "momentum_score": 90,
                "volume_score": 85,
                "sector_score": 86,
                "fund_score": 80,
                "pattern_score": 90,
                "sentiment_score": 82,
                "rating": "A",
                "suggestion": "auction_confirm",
            }
        ]
    )

    path = DailyReportGenerator(config).generate(trade_date="2026-06-04", candidates=candidates)

    content = path.read_text(encoding="utf-8")
    assert "600001.SH" in content
    assert "| stock_code |" in content
