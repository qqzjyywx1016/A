"""Ingest qfq A-share data from baostock into the local standard schema."""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from astock_quant.config.loader import load_config
from astock_quant.data.storage import StorageManager


@dataclass
class Throttle:
    """Pace per-stock baostock requests for long overnight full-market pulls.

    After every request the caller invokes ``tick()``. A short ``sleep_seconds``
    pause is taken between requests; every ``batch_size`` requests a longer
    ``batch_rest_seconds`` pause is taken instead (the "pull a while, rest a
    while" pattern). ``jitter`` randomizes each pause by +-fraction so the
    cadence is not a fixed robotic interval. All-zero defaults make ``tick()``
    a no-op, so small-sample runs behave exactly as before.
    """

    sleep_seconds: float = 0.0
    batch_size: int = 0
    batch_rest_seconds: float = 0.0
    jitter: float = 0.2
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _rand: Callable[[float, float], float] = field(default=random.uniform, repr=False)
    _count: int = field(default=0, repr=False)

    def tick(self) -> None:
        """Account for one completed request and pause according to the policy."""

        self._count += 1
        if self.batch_size > 0 and self.batch_rest_seconds > 0 and self._count % self.batch_size == 0:
            rest = self._with_jitter(self.batch_rest_seconds)
            print(f"batch rest: sleeping {rest:.1f}s after {self._count} requests")
            self._sleep(rest)
        elif self.sleep_seconds > 0:
            self._sleep(self._with_jitter(self.sleep_seconds))

    def _with_jitter(self, base: float) -> float:
        if self.jitter <= 0:
            return base
        delta = base * self.jitter
        return max(0.0, base + self._rand(-delta, delta))

DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,isST"
)
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
INDEX_CODES = {
    "sh.000300": "沪深300",
    "sh.000905": "中证500",
    "sh.000852": "中证1000",
    "sz.399006": "创业板指",
    "sh.000001": "上证指数",
    "sz.399303": "国证2000",
}


def baostock_to_standard_code(code: str) -> str:
    """Convert baostock code like sh.600000 to 600000.SH."""

    market, symbol = str(code).lower().split(".", maxsplit=1)
    return f"{symbol}.{market.upper()}"


def standard_to_baostock_code(code: str) -> str:
    """Convert standard code like 600000.SH to sh.600000."""

    symbol, exchange = str(code).upper().split(".", maxsplit=1)
    return f"{exchange.lower()}.{symbol}"


def exchange_from_stock_code(stock_code: str) -> str:
    """Infer exchange from the standard stock code."""

    symbol = str(stock_code).split(".", maxsplit=1)[0]
    if symbol.startswith(("60", "68")):
        return "SH"
    if symbol.startswith(("00", "30")):
        return "SZ"
    return str(stock_code).split(".")[-1].upper() if "." in str(stock_code) else ""


def is_baostock_stock_code(code: str) -> bool:
    """Return True only for A-share stock codes, excluding indexes and BJ."""

    code = str(code).lower()
    if code.startswith("sh."):
        symbol = code.split(".", maxsplit=1)[1]
        return symbol.startswith(("600", "601", "603", "605", "688"))
    if code.startswith("sz."):
        symbol = code.split(".", maxsplit=1)[1]
        return symbol.startswith(("000", "001", "002", "003", "300", "301"))
    return False


def limit_pct_for_stock(stock_code: str, is_st: bool = False) -> float:
    """Return the daily limit percentage used by the baostock ingest fallback."""

    symbol = str(stock_code).split(".", maxsplit=1)[0]
    if bool(is_st):
        return 0.05
    if symbol.startswith(("30", "688")):
        return 0.20
    return 0.10


def derive_limit_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add daily limit flags from decimal pct_chg plus sealed flags from low/high ratios.

    Ratios to prev_close are invariant under price adjustment, so these flags
    stay correct on qfq data where absolute limit prices cannot be reproduced.
    A sealed limit-up never traded below the limit price all day (low ratio at
    the limit); sealed limit-down likewise via the high ratio.
    """

    result = frame.copy()
    if result.empty:
        for column in ["is_limit_up", "is_limit_down", "is_sealed_limit_up", "is_sealed_limit_down"]:
            result[column] = pd.Series(dtype=bool)
        return result

    pct_chg = pd.to_numeric(result.get("pct_chg"), errors="coerce")
    is_st = result.get("is_st", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    limit_pct = pd.Series(
        [limit_pct_for_stock(code, st) for code, st in zip(result["stock_code"], is_st, strict=False)],
        index=result.index,
    )
    eps = 1e-9
    threshold = limit_pct - 0.001 - eps
    result["is_limit_up"] = pct_chg >= threshold
    result["is_limit_down"] = pct_chg <= -threshold

    nan_series = pd.Series(pd.NA, index=result.index)
    prev_close = pd.to_numeric(result.get("prev_close", nan_series), errors="coerce")
    low = pd.to_numeric(result.get("low", nan_series), errors="coerce")
    high = pd.to_numeric(result.get("high", nan_series), errors="coerce")
    low_ratio = low / prev_close.replace(0, pd.NA) - 1
    high_ratio = high / prev_close.replace(0, pd.NA) - 1
    result["is_sealed_limit_up"] = (result["is_limit_up"] & (low_ratio >= threshold)).fillna(False).astype(bool)
    result["is_sealed_limit_down"] = (result["is_limit_down"] & (high_ratio <= -threshold)).fillna(False).astype(bool)
    return result


def normalize_daily_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Map baostock daily qfq bars to the project daily_bars schema."""

    columns = [
        "stock_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume",
        "turnover_amount",
        "turnover_rate",
        "is_suspended",
        "is_st",
        "pct_chg",
        "adjust_type",
        "is_limit_up",
        "is_limit_down",
        "is_sealed_limit_up",
        "is_sealed_limit_down",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(
        {
            "stock_code": raw["code"].map(baostock_to_standard_code),
            "trade_date": raw["date"].astype(str),
            "open": _numeric(raw.get("open")),
            "high": _numeric(raw.get("high")),
            "low": _numeric(raw.get("low")),
            "close": _numeric(raw.get("close")),
            "prev_close": _numeric(raw.get("preclose")),
            "volume": _numeric(raw.get("volume")),
            "turnover_amount": _numeric(raw.get("amount")),
            "turnover_rate": _numeric(raw.get("turn")),
            "is_suspended": raw.get("tradestatus", pd.Series("1", index=raw.index)).astype(str) != "1",
            "is_st": raw.get("isST", pd.Series("0", index=raw.index)).astype(str) == "1",
            "pct_chg": _numeric(raw.get("pctChg")) / 100.0,
            "adjust_type": "qfq",
        }
    )
    result = derive_limit_flags(result)
    return result[columns].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def normalize_stock_basic(raw: pd.DataFrame) -> pd.DataFrame:
    """Map baostock stock_basic rows without fabricating float market cap."""

    columns = [
        "stock_code",
        "stock_name",
        "exchange",
        "list_date",
        "delist_date",
        "is_delisted",
        "is_st",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(
        {
            "stock_code": raw["code"].map(baostock_to_standard_code),
            "stock_name": raw.get("code_name", pd.Series("", index=raw.index)).fillna("").astype(str),
            "list_date": raw.get("ipoDate", pd.Series("", index=raw.index)).replace("", pd.NA),
            "delist_date": raw.get("outDate", pd.Series("", index=raw.index)).replace("", pd.NA),
            "is_delisted": raw.get("status", pd.Series("1", index=raw.index)).astype(str) == "0",
            "is_st": None,
        }
    )
    result["exchange"] = result["stock_code"].map(exchange_from_stock_code)
    return result[columns].sort_values("stock_code").reset_index(drop=True)


def normalize_sector_map(raw: pd.DataFrame) -> pd.DataFrame:
    """Map baostock industry membership to the standard sector_map schema."""

    columns = ["stock_code", "sector_code", "sector_name", "sector_type"]
    if raw.empty or "code" not in raw.columns:
        return pd.DataFrame(columns=columns)

    industry = raw.get("industry", pd.Series("", index=raw.index)).fillna("").astype(str)
    result = pd.DataFrame(
        {
            "stock_code": raw["code"].map(baostock_to_standard_code),
            "sector_code": industry.replace("", "unknown"),
            "sector_name": industry.replace("", "unknown"),
            "sector_type": "industry",
        }
    )
    result = result[~result["stock_code"].str.startswith(("4", "8"), na=False)]
    return result.drop_duplicates(["stock_code", "sector_code"]).reset_index(drop=True)


def build_sector_daily(daily_bars: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stock daily bars into industry-level daily bars."""

    columns = [
        "sector_code",
        "sector_name",
        "trade_date",
        "close",
        "amount",
        "turnover_amount",
        "sector_stock_count",
        "limit_up_count",
        "sector_type",
    ]
    if daily_bars.empty or sector_map.empty:
        return pd.DataFrame(columns=columns)

    map_cols = ["stock_code", "sector_code", "sector_name", "sector_type"]
    merged = daily_bars.merge(sector_map[map_cols].drop_duplicates(), on="stock_code", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["turnover_amount"] = pd.to_numeric(merged["turnover_amount"], errors="coerce").fillna(0.0)
    merged["close"] = pd.to_numeric(merged["close"], errors="coerce")
    merged["is_limit_up"] = merged.get("is_limit_up", pd.Series(False, index=merged.index)).fillna(False).astype(bool)
    grouped = merged.groupby(["sector_code", "sector_name", "sector_type", "trade_date"], as_index=False)
    result = grouped.apply(_sector_daily_row, include_groups=False).reset_index(drop=True)
    return result[columns].sort_values(["sector_code", "trade_date"]).reset_index(drop=True)


def normalize_index_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Map baostock index daily bars to the project index_bars schema."""

    columns = ["index_code", "trade_date", "open", "high", "low", "close", "amount", "turnover_amount"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            "index_code": raw["code"].map(baostock_to_standard_code),
            "trade_date": raw["date"].astype(str),
            "open": _numeric(raw.get("open")),
            "high": _numeric(raw.get("high")),
            "low": _numeric(raw.get("low")),
            "close": _numeric(raw.get("close")),
            "amount": _numeric(raw.get("amount")),
        }
    )
    result["turnover_amount"] = result["amount"]
    return result[columns].sort_values(["index_code", "trade_date"]).reset_index(drop=True)


def run_ingest(args: argparse.Namespace) -> int:
    """Run the baostock ingestion workflow."""

    import socket

    import baostock as bs  # type: ignore[import-not-found]

    socket.setdefaulttimeout(args.timeout)
    config = load_config(args.config)
    storage = StorageManager(config)
    baostock_codes = _parse_codes(args.codes)
    throttle = Throttle(
        sleep_seconds=args.sleep,
        batch_size=args.batch_size,
        batch_rest_seconds=args.batch_rest,
        jitter=args.jitter,
    )
    failed: dict[str, list[str]] = {"stock_basic": [], "daily": [], "industry": []}
    _login_with_retry(bs)

    try:
        stock_basic_raw = _fetch_stock_basic(
            bs,
            args.include_delisted,
            baostock_codes,
            args.limit,
            start_date=args.start,
            end_date=args.end,
            failed_codes=failed["stock_basic"],
            throttle=throttle,
        )
        stock_basic = normalize_stock_basic(stock_basic_raw)
        storage.save_parquet(stock_basic, "stock_basic.parquet")
        print(f"stock_basic rows={len(stock_basic)}")

        existing_daily = _read_existing_parquet(storage, "daily_bars.parquet")
        daily_bars = _ingest_daily_bars_incremental(
            bs,
            storage,
            stock_basic_raw["code"].tolist(),
            existing_daily,
            args.start,
            args.end,
            args.save_every,
            failed["daily"],
            throttle=throttle,
            relogin_every=args.relogin_every,
        )
        print(f"daily_bars rows={len(daily_bars)}")

        sector_raw = _fetch_sector_map(
            bs, stock_basic_raw["code"].tolist(), failed_codes=failed["industry"], throttle=throttle
        )
        sector_map = normalize_sector_map(sector_raw)
        storage.save_parquet(sector_map, "sector_map.parquet")
        print(f"sector_map rows={len(sector_map)}")

        sector_daily = build_sector_daily(daily_bars, sector_map)
        storage.save_parquet(sector_daily, "sector_daily.parquet")
        print(f"sector_daily rows={len(sector_daily)}")

        index_bars = _fetch_index_bars(bs, args.start, args.end)
        storage.save_parquet(index_bars, "index_bars.parquet")
        print(f"index_bars rows={len(index_bars)}")
        _print_failed_summary(failed)
    finally:
        bs.logout()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest baostock qfq data into data/processed")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    parser.add_argument("--codes", default="", help="Comma-separated baostock or standard stock codes")
    parser.add_argument("--include-delisted", dest="include_delisted", action="store_true", default=True)
    parser.add_argument("--no-include-delisted", dest="include_delisted", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit for number of stocks")
    parser.add_argument("--config", default=None, help="Config path")
    parser.add_argument("--timeout", type=float, default=30, help="Socket timeout seconds for baostock calls")
    parser.add_argument("--save-every", type=int, default=50, help="Persist daily bars after this many successful stocks")
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Seconds to pause between per-stock requests (e.g. 0.5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Take a longer rest after this many requests (0 disables batch rest)",
    )
    parser.add_argument(
        "--batch-rest",
        type=float,
        default=0.0,
        help="Seconds to rest between batches when --batch-size is set (e.g. 60)",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.2,
        help="Randomize each pause by +-this fraction so the cadence is not fixed",
    )
    parser.add_argument(
        "--relogin-every",
        type=int,
        default=0,
        help="Proactively re-login to baostock every N daily requests (0 disables)",
    )
    return parser


def main() -> int:
    return run_ingest(build_arg_parser().parse_args())


def _numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values.replace("", pd.NA), errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _sector_daily_row(group: pd.DataFrame) -> pd.Series:
    amount = group["turnover_amount"].sum()
    if amount > 0:
        close = (group["close"] * group["turnover_amount"]).sum() / amount
    else:
        close = group["close"].mean()
    return pd.Series(
        {
            "sector_code": group.name[0],
            "sector_name": group.name[1],
            "sector_type": group.name[2],
            "trade_date": group.name[3],
            "close": close,
            "amount": amount,
            "turnover_amount": amount,
            "sector_stock_count": int(group["stock_code"].nunique()),
            "limit_up_count": int(group["is_limit_up"].sum()),
        }
    )


def _parse_codes(codes: str) -> list[str]:
    parsed = [code.strip() for code in str(codes or "").split(",") if code.strip()]
    result = []
    for code in parsed:
        if "." in code and code.split(".", maxsplit=1)[0].lower() in {"sh", "sz", "bj"}:
            result.append(code.lower())
        else:
            result.append(standard_to_baostock_code(code))
    return result


def _login_with_retry(bs: Any, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> Any:
    last_error = ""
    for attempt in range(1, retries + 1):
        result = bs.login()
        error_code = getattr(result, "error_code", "0")
        error_msg = getattr(result, "error_msg", "")
        if error_code == "0":
            return result
        last_error = f"{error_code} {error_msg}".strip()
        print(f"baostock login attempt {attempt}/{retries} failed: {last_error}")
        if attempt < retries:
            sleep(min(2**attempt, 10))
    raise RuntimeError(f"baostock login failed after {retries} attempts: {last_error}")


# baostock keeps one long-lived socket; these substrings flag the receive/reset
# errors that mean that socket is dead and only a re-login (not a plain retry)
# can recover it.
_CONNECTION_ERROR_HINTS = (
    "10054",
    "10053",
    "10060",
    "10002007",
    "网络接收",
    "接收数据异常",
    "发送数据异常",
    "强迫关闭",
    "网络连接",
    "connection",
    "reset",
    "timed out",
    "broken pipe",
    "recv",
)


def _looks_like_connection_error(error: object) -> bool:
    """Return True when the error text looks like a dropped baostock socket."""

    text = str(error).lower()
    return any(hint.lower() in text for hint in _CONNECTION_ERROR_HINTS)


def _relogin(bs: Any, sleep: Callable[[float], None] | None = None) -> None:
    """Drop and re-establish the baostock session after a connection error.

    Retrying a query on the same dead socket fails forever (the cascade of
    receive errors seen on long full-market pulls). Logging out and back in
    gives subsequent queries a fresh socket.
    """

    do_sleep = sleep if sleep is not None else time.sleep
    try:
        bs.logout()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        print(f"relogin: logout failed (ignored): {exc}")
    do_sleep(2.0)
    _login_with_retry(bs, sleep=do_sleep)
    print("relogin: re-established baostock session")


def _proactive_relogin(bs: Any, processed: int, relogin_every: int) -> None:
    """Refresh the session every ``relogin_every`` requests, before it goes stale."""

    if bs is None or relogin_every <= 0 or processed <= 0 or processed % relogin_every != 0:
        return
    print(f"proactive relogin after {processed} requests to refresh the baostock session")
    try:
        _relogin(bs)
    except Exception as exc:  # pragma: no cover - live path
        print(f"proactive relogin failed: {exc}")


def _query_result(
    factory: Callable[..., Any],
    label: str,
    *args: Any,
    retries: int = 3,
    bs: Any | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"{label} start attempt {attempt}/{retries}")
            rs = factory(*args, **kwargs)
            error_code = getattr(rs, "error_code", "0")
            if error_code != "0":
                raise RuntimeError(f"{label} failed: {error_code} {getattr(rs, 'error_msg', '')}")
            rows: list[list[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())
            result = pd.DataFrame(rows, columns=getattr(rs, "fields", []))
            print(f"{label} done rows={len(result)}")
            return result
        except Exception as exc:  # pragma: no cover - live retry path
            last_error = exc
            print(f"{label} attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
                # On a dead socket a plain retry is hopeless; re-login first.
                if bs is not None and _looks_like_connection_error(exc):
                    print(f"{label}: connection looks dead, re-logging into baostock before retry")
                    try:
                        _relogin(bs)
                    except Exception as relogin_exc:  # pragma: no cover - live path
                        print(f"{label}: relogin failed: {relogin_exc}")
    raise RuntimeError(f"{label} failed after {retries} attempts") from last_error


def _fetch_stock_basic(
    bs: Any,
    include_delisted: bool,
    codes: list[str],
    limit: int | None,
    start_date: str | None = None,
    end_date: str | None = None,
    failed_codes: list[str] | None = None,
    retries: int = 3,
    throttle: Throttle | None = None,
) -> pd.DataFrame:
    selected_codes = list(dict.fromkeys(codes))
    if not selected_codes:
        if end_date is None:
            raise ValueError("end_date is required when --codes is empty")
        selected_codes = _enumerate_stock_codes(bs, start_date, end_date, include_delisted)
    selected_codes = [code for code in selected_codes if is_baostock_stock_code(code)]
    if limit is not None:
        selected_codes = selected_codes[:limit]

    frames = []
    for idx, code in enumerate(selected_codes, start=1):
        print(f"[{idx}/{len(selected_codes)}] stock basic {code}")
        try:
            frame = _query_result(bs.query_stock_basic, f"stock basic {code}", retries=retries, bs=bs, code=code)
        except RuntimeError as exc:
            print(f"WARNING stock basic {code} skipped: {exc}")
            if failed_codes is not None:
                failed_codes.append(code)
            if throttle is not None:
                throttle.tick()
            continue
        if not frame.empty:
            frames.append(frame)
        if throttle is not None:
            throttle.tick()

    if not frames:
        return pd.DataFrame(columns=["code", "code_name", "ipoDate", "outDate", "type", "status"])
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        return raw
    raw = raw[raw.get("type", "1").astype(str) == "1"].copy()
    if include_delisted:
        raw = raw[raw.get("status", "1").astype(str).isin(["0", "1"])].copy()
    else:
        raw = raw[raw.get("status", "1").astype(str) == "1"].copy()
    raw = raw[~raw["code"].astype(str).str.lower().str.startswith("bj.")].copy()
    return raw.drop_duplicates("code").reset_index(drop=True)


def _enumerate_stock_codes(
    bs: Any,
    start_date: str | None,
    end_date: str,
    include_delisted: bool,
    today: pd.Timestamp | None = None,
) -> list[str]:
    # baostock has no data for future dates, so never enumerate past today: a
    # future --end (or current-year year-end) would otherwise return empty rows
    # and crash the whole run before any stock is fetched.
    today_ts = (today if today is not None else pd.Timestamp.today()).normalize()
    end_ts = min(pd.Timestamp(end_date).normalize(), today_ts)
    dates = [end_ts.strftime("%Y-%m-%d")]
    if include_delisted and start_date:
        start_ts = pd.Timestamp(start_date).normalize()
        for year in range(start_ts.year, end_ts.year + 1):
            year_end = min(pd.Timestamp(year=year, month=12, day=31), today_ts)
            if start_ts <= year_end <= end_ts:
                dates.append(year_end.strftime("%Y-%m-%d"))
    dates = list(dict.fromkeys(dates))

    codes: list[str] = []
    seen: set[str] = set()
    for query_date in dates:
        # One bad enumeration date (holiday gap, transient failure) must not kill
        # the run as long as another date yields the universe.
        try:
            frame = _query_all_stock_with_fallback(bs, query_date, today=today_ts)
        except RuntimeError as exc:
            print(f"WARNING enumerate: no data near {query_date}, skipping: {exc}")
            continue
        if frame.empty or "code" not in frame.columns:
            continue
        for code in frame["code"].dropna().astype(str):
            code = code.lower()
            if not is_baostock_stock_code(code) or code in seen:
                continue
            seen.add(code)
            codes.append(code)
    if not codes:
        raise RuntimeError(
            f"query_all_stock returned no codes for any of {dates}; "
            "check that --end is a past trading day and baostock is reachable"
        )
    return codes


def _query_all_stock_with_fallback(
    bs: Any, day: str, max_backtrack_days: int = 20, today: pd.Timestamp | None = None
) -> pd.DataFrame:
    today_ts = (today if today is not None else pd.Timestamp.today()).normalize()
    # Start the backtrack from today at the latest; a future day has no data.
    current = min(pd.Timestamp(day).normalize(), today_ts)
    last_error: RuntimeError | None = None
    for offset in range(max_backtrack_days + 1):
        query_day = (current - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            frame = _query_result(bs.query_all_stock, f"all stock {query_day}", bs=bs, day=query_day)
        except RuntimeError as exc:
            last_error = exc
            continue
        if not frame.empty:
            if query_day != pd.Timestamp(day).normalize().strftime("%Y-%m-%d"):
                print(f"query_all_stock fallback {day} -> {query_day}")
            return frame
    raise RuntimeError(f"query_all_stock found no data near {day}") from last_error


def _fetch_sector_map(
    bs: Any,
    codes: list[str],
    failed_codes: list[str] | None = None,
    retries: int = 3,
    throttle: Throttle | None = None,
) -> pd.DataFrame:
    rows = []
    fields: list[str] | None = None
    for idx, code in enumerate(codes, start=1):
        print(f"[{idx}/{len(codes)}] stock industry {code}")
        try:
            frame = _query_result(bs.query_stock_industry, f"stock industry {code}", retries=retries, bs=bs, code=code)
        except RuntimeError as exc:
            print(f"stock industry {code} fallback unknown: {exc}")
            if failed_codes is not None:
                failed_codes.append(code)
            rows.append({"code": code, "industry": "unknown"})
            if throttle is not None:
                throttle.tick()
            continue
        if frame.empty:
            rows.append({"code": code, "industry": "unknown"})
            if throttle is not None:
                throttle.tick()
            continue
        if fields is None:
            fields = list(frame.columns)
        rows.extend(frame.to_dict("records"))
        if throttle is not None:
            throttle.tick()
    if not rows:
        return pd.DataFrame(columns=["code", "industry"])
    result = pd.DataFrame(rows)
    if "code" not in result.columns:
        result["code"] = codes[: len(result)]
    if "industry" not in result.columns:
        result["industry"] = "unknown"
    return result


def _query_history(
    bs: Any,
    code: str,
    fields: str,
    start_date: str,
    end_date: str,
    *,
    adjustflag: str,
    label: str,
    retries: int = 3,
) -> pd.DataFrame:
    return _query_result(
        bs.query_history_k_data_plus,
        label,
        code,
        fields,
        retries=retries,
        bs=bs,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )


def _ingest_daily_bars_incremental(
    bs: Any,
    storage: StorageManager,
    codes: list[str],
    existing_daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    save_every: int,
    failed_codes: list[str],
    retries: int = 3,
    throttle: Throttle | None = None,
    relogin_every: int = 0,
) -> pd.DataFrame:
    persisted = _merge_existing_daily(existing_daily, [])
    fetched_daily: list[pd.DataFrame] = []
    successful_since_save = 0
    requests_made = 0
    flush_every = max(int(save_every or 0), 1)

    for idx, code in enumerate(codes, start=1):
        standard_code = baostock_to_standard_code(code)
        if _existing_code_covers_range(persisted, standard_code, start_date, end_date):
            print(f"[{idx}/{len(codes)}] skip {code} existing coverage")
            continue
        try:
            raw_bars = _query_history(
                bs,
                code,
                DAILY_FIELDS,
                start_date,
                end_date,
                adjustflag="2",
                label=f"daily {code}",
                retries=retries,
            )
        except RuntimeError as exc:
            print(f"WARNING daily {code} skipped: {exc}")
            failed_codes.append(code)
            if throttle is not None:
                throttle.tick()
            requests_made += 1
            _proactive_relogin(bs, requests_made, relogin_every)
            continue
        normalized = normalize_daily_bars(raw_bars)
        fetched_daily.append(normalized)
        successful_since_save += 1
        print(f"[{idx}/{len(codes)}] {code} daily rows={len(normalized)}")
        if throttle is not None:
            throttle.tick()
        requests_made += 1
        _proactive_relogin(bs, requests_made, relogin_every)
        if successful_since_save >= flush_every:
            persisted = _merge_existing_daily(persisted, fetched_daily)
            storage.save_parquet(persisted, "daily_bars.parquet")
            fetched_daily = []
            successful_since_save = 0
            print(f"daily_bars checkpoint rows={len(persisted)}")

    persisted = _merge_existing_daily(persisted, fetched_daily)
    storage.save_parquet(persisted, "daily_bars.parquet")
    print(f"daily_bars final checkpoint rows={len(persisted)}")
    return persisted


def _fetch_index_bars(bs: Any, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for code, name in INDEX_CODES.items():
        try:
            raw = _query_history(
                bs,
                code,
                INDEX_FIELDS,
                start_date,
                end_date,
                adjustflag="3",
                label=f"index {code}",
            )
        except RuntimeError as exc:
            if code == "sz.399303":
                print(f"国证2000不可得,规模基准用中证1000: {exc}")
                continue
            raise
        normalized = normalize_index_bars(raw)
        normalized["index_name"] = name
        frames.append(normalized)
        print(f"index {code} rows={len(normalized)}")
    if not frames:
        return pd.DataFrame(columns=["index_code", "trade_date", "open", "high", "low", "close", "amount"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["index_code", "trade_date"])


def _read_existing_parquet(storage: StorageManager, file_name: str) -> pd.DataFrame:
    path = storage.processed_path / file_name
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _existing_code_covers_range(existing: pd.DataFrame, stock_code: str, start_date: str, end_date: str) -> bool:
    if existing.empty or "stock_code" not in existing.columns or "trade_date" not in existing.columns:
        return False
    subset = existing[existing["stock_code"] == stock_code]
    if subset.empty:
        return False
    dates = pd.to_datetime(subset["trade_date"], errors="coerce")
    return bool(dates.min() <= pd.Timestamp(start_date) and dates.max() >= pd.Timestamp(end_date))


def _merge_existing_daily(existing: pd.DataFrame, fetched: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in [existing, *fetched] if not frame.empty]
    if not frames:
        return normalize_daily_bars(pd.DataFrame())
    merged = pd.concat(frames, ignore_index=True)
    if "trade_date" in merged.columns:
        original = merged["trade_date"].astype(str)
        parsed = pd.to_datetime(merged["trade_date"], errors="coerce")
        merged["trade_date"] = parsed.dt.strftime("%Y-%m-%d")
        merged.loc[parsed.isna(), "trade_date"] = original[parsed.isna()]
    return merged.drop_duplicates(["stock_code", "trade_date"], keep="last").sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def _print_failed_summary(failed: dict[str, list[str]]) -> None:
    for label, codes in failed.items():
        if codes:
            print(f"WARNING failed {label} count={len(codes)} codes={','.join(codes)}")


if __name__ == "__main__":
    raise SystemExit(main())
