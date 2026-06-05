"""Markdown daily report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from astock_quant.config.loader import resolve_path


class DailyReportGenerator:
    """Generate daily Markdown reports for market state and selected stocks."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.report_path = resolve_path(config, config.get("data", {}).get("report_path", "reports"))
        self.report_path.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        trade_date: str,
        candidates: pd.DataFrame,
        market_state: dict[str, Any] | None = None,
        sector_rank: pd.DataFrame | None = None,
    ) -> Path:
        """Write reports/YYYY-MM-DD_daily_report.md and return its path."""

        market_state = market_state or {}
        sector_rank = sector_rank if sector_rank is not None else pd.DataFrame()
        path = self.report_path / f"{trade_date}_daily_report.md"
        lines = [
            f"# {trade_date} AStock 每日选股报告",
            "",
            "## 1. 今日市场状态",
            f"- 市场状态：{market_state.get('market_regime', 'neutral')}",
            f"- 上涨比例：{market_state.get('market_up_ratio', 'N/A')}",
            f"- 涨停数：{market_state.get('limit_up_count', 'N/A')}",
            f"- 跌停数：{market_state.get('limit_down_count', 'N/A')}",
            "",
            "## 2. 情绪周期判断",
            self._sentiment_text(str(market_state.get("market_regime", "neutral"))),
            "",
            "## 3. 强势板块排名",
            self._table_or_empty(sector_rank),
            "",
            "## 4. 候选股池 Top 20",
            self._candidate_table(candidates),
            "",
            "## 5. 核心候选股说明",
        ]
        core = candidates[candidates.get("rating", pd.Series(dtype=str)) == "A"].head(5) if not candidates.empty else pd.DataFrame()
        if core.empty:
            lines.append("- 今日无 A 级核心候选股。")
        else:
            for _, row in core.iterrows():
                sector_name = row.get("active_sector_name") or row.get("sector", "")
                lines.append(
                    f"- {row.get('stock_code', '')} {row.get('stock_name', '')}："
                    f"总分 {row.get('total_score', '')}，板块 {sector_name}，建议 {row.get('suggestion', 'watch')}。"
                )
        lines.extend(
            [
                "",
                "## 6. 风险提示",
                "- 短线策略受隔夜消息、流动性、涨跌停和高开低走影响较大。",
                "- 候选股仅用于研究和交易计划，不构成自动下单或投资建议。",
                "",
                "## 7. 明日交易计划",
                "- 优先观察 A 级核心池，回避一字涨停和高开超过 7% 的标的。",
                "- 只在竞价确认、盘中突破放量或均线回踩缩量不破时考虑执行。",
                "- 单票仓位不得超过配置上限，触发止损、止盈或持仓天数规则时执行卖出计划。",
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def _sentiment_text(regime: str) -> str:
        mapping = {
            "strong": "市场处于强势接力阶段，可适度提高核心池关注度。",
            "neutral": "市场情绪中性，优先选择板块共振且量价确认的标的。",
            "weak": "市场偏弱，控制总仓位并降低追高频率。",
            "risk_off": "市场风险偏好较低，以防守和空仓观察为主。",
        }
        return mapping.get(regime, mapping["neutral"])

    @staticmethod
    def _table_or_empty(df: pd.DataFrame) -> str:
        if df.empty:
            return "暂无板块排名数据。"
        return DailyReportGenerator._markdown_table(df.head(10))

    @staticmethod
    def _candidate_table(candidates: pd.DataFrame) -> str:
        columns = [
            "stock_code",
            "stock_name",
            "sector",
            "active_sector_name",
            "active_sector_type",
            "total_score",
            "momentum_score",
            "volume_score",
            "sector_score",
            "fund_score",
            "pattern_score",
            "sentiment_score",
            "rps_5",
            "rps_10",
            "rps_20",
            "rps_60",
            "rps_composite",
            "rps_pattern",
            "sector_rps_3",
            "sector_rps_5",
            "sector_rps_10",
            "sector_rps_20",
            "sector_rps_composite",
            "sector_rps_pattern",
            "rating",
            "suggestion",
        ]
        if candidates.empty:
            return "今日无候选股。"
        table = candidates.copy()
        if "active_sector_name" in table.columns:
            table["sector"] = table["active_sector_name"].where(table["active_sector_name"].notna(), table.get("sector", ""))
        for column in columns:
            if column not in table.columns:
                table[column] = ""
        return DailyReportGenerator._markdown_table(table[columns].head(20))

    @staticmethod
    def _markdown_table(df: pd.DataFrame) -> str:
        """Render a small DataFrame as Markdown without optional dependencies."""

        if df.empty:
            return ""
        columns = list(df.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        rows = []
        for _, row in df.iterrows():
            values = [DailyReportGenerator._format_cell(row[column]) for column in columns]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *rows])

    @staticmethod
    def _format_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")
