#!/usr/bin/env python3
"""Validate local real-data readiness before running backtests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astock_quant.config.loader import load_config
from astock_quant.data.astock_data_adapter import AStockDataAdapter
from astock_quant.data.storage import StorageManager


@dataclass(slots=True)
class ValidationOptions:
    """CLI/configurable thresholds for data validation."""

    data_mode: str = "local"
    max_core_missing_rate: float = 0.02
    max_ohlc_violation_rate: float = 0.0
    pct_chg_tolerance: float = 0.001
    delisted_list: str | None = None


@dataclass(slots=True)
class ValidationResult:
    """Structured validation outcome."""

    exit_code: int
    summary: dict[str, Any]


def run_validation(
    config: dict[str, Any],
    start: str,
    end: str,
    options: ValidationOptions,
) -> ValidationResult:
    """Run local data validation and write report artifacts."""

    config = _force_local_mode(config)
    storage = StorageManager(config)
    adapter = AStockDataAdapter(config, storage=storage)
    summary: dict[str, Any] = {
        "start": start,
        "end": end,
        "data_mode": options.data_mode,
        "hard_failed": False,
        "hard_failures": [],
        "warnings": [],
    }

    if options.data_mode != "local":
        _hard(summary, f"unsupported data_mode={options.data_mode}; only local is implemented")
    stock_basic = adapter.get_stock_basic()
    daily_bars = adapter.get_daily_bars(start, end)
    index_bars = adapter.get_index_bars(start, end)

    _check_empty_tables(summary, stock_basic, daily_bars, index_bars)
    if not daily_bars.empty:
        daily_bars = daily_bars.copy()
        daily_bars["trade_date"] = pd.to_datetime(daily_bars["trade_date"]).dt.normalize()
        _check_core_missing(summary, daily_bars, options.max_core_missing_rate)
        _check_qfq(summary, daily_bars)
        _check_ohlc(summary, daily_bars, options.max_ohlc_violation_rate)
        _check_pct_chg(summary, daily_bars, options.pct_chg_tolerance)
        _check_limits(summary, daily_bars)
        _check_float_market_cap(summary, stock_basic, daily_bars)
        _check_pit_universe(summary, stock_basic)
        _check_survivorship(summary, stock_basic, daily_bars, options.delisted_list)
    else:
        summary["qfq_confirmed"] = False
        summary["survivorship_exposure_pct"] = None
        summary["float_market_cap_missing_rate"] = None

    summary["hard_failed"] = bool(summary["hard_failures"])
    _write_outputs(storage, summary)
    return ValidationResult(exit_code=1 if summary["hard_failed"] else 0, summary=summary)


def _force_local_mode(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    external = dict(result.get("external", {}))
    external["live_enabled"] = False
    result["external"] = external
    return result


def _hard(summary: dict[str, Any], message: str) -> None:
    summary["hard_failures"].append(message)


def _warn(summary: dict[str, Any], message: str) -> None:
    summary["warnings"].append(message)


def _check_empty_tables(
    summary: dict[str, Any],
    stock_basic: pd.DataFrame,
    daily_bars: pd.DataFrame,
    index_bars: pd.DataFrame,
) -> None:
    counts = {
        "stock_basic_rows": int(len(stock_basic)),
        "daily_bars_rows": int(len(daily_bars)),
        "index_bars_rows": int(len(index_bars)),
    }
    summary.update(counts)
    if stock_basic.empty:
        _hard(summary, "stock_basic is empty")
    if daily_bars.empty:
        _hard(summary, "daily_bars is empty")
    if index_bars.empty:
        _hard(summary, "index_bars is empty")


def _check_core_missing(summary: dict[str, Any], daily_bars: pd.DataFrame, threshold: float) -> None:
    core_columns = ["stock_code", "trade_date", "open", "high", "low", "close"]
    volume_col = _first_existing(daily_bars, ["volume", "turnover_volume"])
    amount_col = _first_existing(daily_bars, ["amount", "turnover_amount"])
    if volume_col is None:
        _hard(summary, "core volume column missing: expected volume or turnover_volume")
    else:
        core_columns.append(volume_col)
    if amount_col is None:
        _hard(summary, "core amount column missing: expected amount or turnover_amount")
    else:
        core_columns.append(amount_col)

    missing_rates = {}
    for column in core_columns:
        if column not in daily_bars.columns:
            missing_rates[column] = 1.0
            _hard(summary, f"core field {column} missing")
            continue
        rate = float(daily_bars[column].isna().mean())
        missing_rates[column] = round(rate, 6)
        if rate > threshold:
            _hard(summary, f"core field {column} missing_rate {rate:.4f} exceeds {threshold:.4f}")
    summary["core_missing_rates"] = missing_rates


def _check_qfq(summary: dict[str, Any], daily_bars: pd.DataFrame) -> None:
    marker_col = _first_existing(daily_bars, ["adjust_type", "adj_type", "adjust", "price_adjust"])
    if marker_col is None:
        summary["qfq_confirmed"] = False
        _hard(summary, "未确认复权口径,禁止回测")
        return
    values = daily_bars[marker_col].dropna().astype(str).str.lower().str.strip()
    qfq_values = {"qfq", "forward", "front", "前复权"}
    confirmed = bool(not values.empty and values.isin(qfq_values).all())
    summary["qfq_confirmed"] = confirmed
    summary["qfq_marker_column"] = marker_col
    if not confirmed:
        _hard(summary, "未确认复权口径,禁止回测")


def _check_ohlc(summary: dict[str, Any], daily_bars: pd.DataFrame, threshold: float) -> None:
    required = {"open", "high", "low", "close"}
    if not required.issubset(daily_bars.columns):
        summary["ohlc_violation_rate"] = 1.0
        return
    open_ = pd.to_numeric(daily_bars["open"], errors="coerce")
    high = pd.to_numeric(daily_bars["high"], errors="coerce")
    low = pd.to_numeric(daily_bars["low"], errors="coerce")
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    violation = (high < pd.concat([open_, close], axis=1).max(axis=1)) | (
        low > pd.concat([open_, close], axis=1).min(axis=1)
    )
    volume_col = _first_existing(daily_bars, ["volume", "turnover_volume"])
    amount_col = _first_existing(daily_bars, ["amount", "turnover_amount"])
    if volume_col is not None:
        violation |= pd.to_numeric(daily_bars[volume_col], errors="coerce") < 0
    if amount_col is not None:
        violation |= pd.to_numeric(daily_bars[amount_col], errors="coerce") < 0
    rate = float(violation.fillna(True).mean())
    summary["ohlc_violation_rate"] = round(rate, 6)
    if rate > threshold:
        _hard(summary, f"OHLC/volume/amount violation rate {rate:.4f} exceeds {threshold:.4f}")


def _check_pct_chg(summary: dict[str, Any], daily_bars: pd.DataFrame, tolerance: float) -> None:
    pre_col = _first_existing(daily_bars, ["prev_close", "pre_close"])
    if pre_col is None or "pct_chg" not in daily_bars.columns:
        _warn(summary, "pct_chg/pre_close unavailable; pct_chg consistency check skipped")
        return
    expected = pd.to_numeric(daily_bars["close"], errors="coerce") / pd.to_numeric(daily_bars[pre_col], errors="coerce") - 1
    actual = pd.to_numeric(daily_bars["pct_chg"], errors="coerce")
    bad = (actual - expected).abs() > tolerance
    rate = float(bad.dropna().mean()) if bad.notna().any() else 0.0
    summary["pct_chg_deviation_over_threshold_rate"] = round(rate, 6)
    if rate > 0:
        _warn(summary, f"pct_chg deviation over tolerance rate={rate:.4f}")


def _check_limits(summary: dict[str, Any], daily_bars: pd.DataFrame) -> None:
    pre_col = _first_existing(daily_bars, ["prev_close", "pre_close"])
    if pre_col is None:
        _warn(summary, "prev_close/pre_close unavailable; limit status derivation skipped")
        return
    data = daily_bars.copy()
    up_prices = []
    down_prices = []
    for _, row in data.iterrows():
        up, down = _limit_prices(str(row.get("stock_code")), row)
        up_prices.append(up)
        down_prices.append(down)
    data["derived_limit_up_price"] = up_prices
    data["derived_limit_down_price"] = down_prices
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    limit_up_price = pd.Series(up_prices, index=data.index)
    limit_down_price = pd.Series(down_prices, index=data.index)
    is_limit_up = close >= limit_up_price - 1e-6
    is_limit_down = close <= limit_down_price + 1e-6
    sealed_up = is_limit_up & (low >= limit_up_price - 1e-6)
    sealed_down = is_limit_down & (high <= limit_down_price + 1e-6)
    fried_board = (high >= limit_up_price - 1e-6) & ~is_limit_up
    summary["limit_status"] = {
        "derived_limit_up_count": int(is_limit_up.sum()),
        "derived_limit_down_count": int(is_limit_down.sum()),
        "sealed_limit_up_count": int(sealed_up.sum()),
        "sealed_limit_down_count": int(sealed_down.sum()),
        "fried_board_daily_approx_count": int(fried_board.sum()),
        "note": "炸板(曾涨停未封)由日线近似,high触板&close未封,非真实盘口",
    }


def _check_float_market_cap(summary: dict[str, Any], stock_basic: pd.DataFrame, daily_bars: pd.DataFrame) -> None:
    source = "unknown"
    point_in_time = False
    for frame in [daily_bars, stock_basic]:
        if "float_market_cap_source" in frame.columns and frame["float_market_cap_source"].notna().any():
            source = str(frame["float_market_cap_source"].dropna().iloc[0])
        if "float_market_cap_is_point_in_time" in frame.columns:
            point_in_time = bool(frame["float_market_cap_is_point_in_time"].fillna(False).astype(bool).any())
    cap_frame = daily_bars if "float_market_cap" in daily_bars.columns else stock_basic
    if "float_market_cap" in cap_frame.columns:
        caps = pd.to_numeric(cap_frame["float_market_cap"], errors="coerce")
        missing_rate = float(caps.isna().mean())
    else:
        missing_rate = 1.0
        _warn(summary, "float_market_cap missing; market-cap filters may be unavailable")
    jump_count = 0
    if {"stock_code", "trade_date", "float_market_cap"}.issubset(daily_bars.columns):
        data = daily_bars.sort_values(["stock_code", "trade_date"]).copy()
        jumps = data.groupby("stock_code")["float_market_cap"].pct_change().abs() > 0.5
        jump_count = int(jumps.fillna(False).sum())
    summary.update(
        {
            "float_market_cap_source": source,
            "float_market_cap_is_point_in_time": point_in_time,
            "float_market_cap_missing_rate": round(missing_rate, 6),
            "float_market_cap_suspicious_jump_count": jump_count,
        }
    )
    if not point_in_time:
        _warn(summary, "float_market_cap may not be point-in-time")


def _check_pit_universe(summary: dict[str, Any], stock_basic: pd.DataFrame) -> None:
    pit_markers = ["is_st_source", "list_date_source", "delist_status_source"]
    marker_values = []
    for column in pit_markers:
        if column in stock_basic.columns:
            marker_values.extend(stock_basic[column].dropna().astype(str).str.lower().tolist())
    is_pit = bool(marker_values and all(value in {"as_of", "pit", "point_in_time"} for value in marker_values))
    summary["universe_metadata_is_point_in_time"] = is_pit
    if not is_pit:
        _warn(summary, "universe 过滤存在 is_st/市值的反向 look-ahead")


def _check_survivorship(
    summary: dict[str, Any],
    stock_basic: pd.DataFrame,
    daily_bars: pd.DataFrame,
    delisted_list: str | None,
) -> None:
    universe_codes = set(stock_basic.get("stock_code", pd.Series(dtype=str)).dropna().astype(str))
    universe_codes |= set(daily_bars.get("stock_code", pd.Series(dtype=str)).dropna().astype(str))
    if not delisted_list:
        summary["survivorship_exposure_pct"] = None
        _warn(summary, "仅当前在册股票,结果为乐观上界")
        return
    path = Path(delisted_list)
    if not path.exists():
        summary["survivorship_exposure_pct"] = None
        _warn(summary, f"delisted list not found: {delisted_list}")
        return
    listed = pd.read_csv(path)
    if "stock_code" not in listed.columns:
        summary["survivorship_exposure_pct"] = None
        _warn(summary, "delisted list missing stock_code column")
        return
    risky_codes = set(listed["stock_code"].dropna().astype(str))
    exposure = len(universe_codes & risky_codes) / len(universe_codes) if universe_codes else 0.0
    summary["survivorship_exposure_pct"] = round(exposure * 100, 6)


def _write_outputs(storage: StorageManager, summary: dict[str, Any]) -> None:
    summary_path = storage.result_path / "data_validation_summary.json"
    report_path = storage.report_path / "data_validation_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 数据偏差/局限",
        "",
        "本报告把数据偏差和局限置于最前面。若仅当前在册股票可用，结果为乐观上界；尤其会高估小盘动量。板块后见之明、规模 beta、复权口径和逐日流通市值口径会直接影响回测可用性。",
        "",
        "## 阻止回测的硬失败",
    ]
    hard = summary.get("hard_failures", [])
    lines.extend([f"- {item}" for item in hard] or ["- 无"])
    lines.extend(["", "## 软风险与警告"])
    lines.extend([f"- {item}" for item in summary.get("warnings", [])] or ["- 无"])
    lines.extend(
        [
            "",
            "## 数据质量摘要",
            "",
            f"- stock_basic_rows: {summary.get('stock_basic_rows')}",
            f"- daily_bars_rows: {summary.get('daily_bars_rows')}",
            f"- index_bars_rows: {summary.get('index_bars_rows')}",
            f"- qfq_confirmed: {summary.get('qfq_confirmed')}",
            f"- ohlc_violation_rate: {summary.get('ohlc_violation_rate')}",
            f"- float_market_cap_source: {summary.get('float_market_cap_source')}",
            f"- float_market_cap_is_point_in_time: {summary.get('float_market_cap_is_point_in_time')}",
            f"- float_market_cap_missing_rate: {summary.get('float_market_cap_missing_rate')}",
            f"- float_market_cap_suspicious_jump_count: {summary.get('float_market_cap_suspicious_jump_count')}",
            f"- survivorship_exposure_pct: {summary.get('survivorship_exposure_pct')}",
            "",
            "## 涨跌停日线近似说明",
            "",
            "炸板(曾涨停未封)由日线近似,high触板&close未封,非真实盘口。",
            "",
            "```json",
            json.dumps(summary.get("limit_status", {}), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _limit_prices(code: str, row: pd.Series) -> tuple[float, float]:
    explicit_up = _to_float(row.get("limit_up_price"))
    explicit_down = _to_float(row.get("limit_down_price"))
    prev_close = _to_float(row.get("prev_close"))
    if prev_close is None:
        prev_close = _to_float(row.get("pre_close"))
    if prev_close is None or prev_close <= 0:
        return explicit_up or float("inf"), explicit_down or 0.0
    limit_pct = _limit_pct(code, row)
    return explicit_up or round(prev_close * (1 + limit_pct), 2), explicit_down or round(prev_close * (1 - limit_pct), 2)


def _limit_pct(code: str, row: pd.Series) -> float:
    is_st = row.get("is_st", False)
    if _truthy(is_st):
        return 0.05
    prefix = str(code).split(".")[0]
    if prefix.startswith("30") or prefix.startswith("688"):
        return 0.20
    if prefix.startswith("8") or prefix.startswith("4"):
        return 0.30
    return 0.10


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "是"}:
            return True
        if normalized in {"0", "false", "no", "n", "否", ""}:
            return False
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-mode", default="local")
    parser.add_argument("--max-core-missing-rate", type=float, default=0.02)
    parser.add_argument("--max-ohlc-violation-rate", type=float, default=0.0)
    parser.add_argument("--pct-chg-tolerance", type=float, default=0.001)
    parser.add_argument("--delisted-list")
    args = parser.parse_args()

    options = ValidationOptions(
        data_mode=args.data_mode,
        max_core_missing_rate=args.max_core_missing_rate,
        max_ohlc_violation_rate=args.max_ohlc_violation_rate,
        pct_chg_tolerance=args.pct_chg_tolerance,
        delisted_list=args.delisted_list,
    )
    result = run_validation(load_config(), args.start, args.end, options)
    if result.exit_code:
        print("BLOCKED: data validation hard failures")
        for issue in result.summary["hard_failures"]:
            print(f"- {issue}")
    else:
        print("OK: data validation passed hard gates")
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
