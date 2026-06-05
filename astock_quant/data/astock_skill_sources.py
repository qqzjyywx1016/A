"""Live data source wrappers extracted from external/a-stock-data/SKILL.md."""

from __future__ import annotations

import logging
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from astock_quant.config.loader import resolve_path


class AStockSkillSource:
    """Fetch live A-share data through endpoints documented by a-stock-data."""

    UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    BAIDU_KLINE_URL = "https://finance.pae.baidu.com/selfselect/getstockquotation"
    TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    EASTMONEY_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
    EASTMONEY_SLIST_URL = "https://push2.eastmoney.com/api/qt/slist/get"
    EASTMONEY_STOCK_INFO_URL = "https://push2.eastmoney.com/api/qt/stock/get"
    EASTMONEY_FUND_FLOW_DAY_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"

    INDEX_SYMBOLS = {
        "000300.SH": {"code": "000300", "name": "沪深300"},
        "000852.SH": {"code": "000852", "name": "中证1000"},
        "000001.SH": {"code": "000001", "name": "上证指数"},
        "399001.SZ": {"code": "399001", "name": "深证成指"},
        "399006.SZ": {"code": "399006", "name": "创业板指"},
    }

    STOCK_BASIC_COLUMNS = [
        "stock_code",
        "stock_name",
        "exchange",
        "market",
        "is_st",
        "is_suspended",
        "list_date",
        "listing_days",
        "sector",
        "total_market_cap",
        "float_market_cap",
    ]
    DAILY_COLUMNS = [
        "stock_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume",
        "amount",
        "turnover_amount",
        "pct_chg",
        "is_suspended",
    ]
    INDEX_COLUMNS = [
        "index_code",
        "index_name",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume",
        "amount",
        "turnover_amount",
        "pct_chg",
    ]
    SECTOR_MAP_COLUMNS = ["stock_code", "sector", "sector_code", "sector_change_pct", "lead_stock", "concept_tags"]
    SECTOR_DAILY_COLUMNS = [
        "sector",
        "trade_date",
        "close",
        "sector_return_1d",
        "sector_return_3d",
        "turnover_amount",
        "up_count",
        "down_count",
        "leader",
    ]
    FUND_FLOW_COLUMNS = [
        "stock_code",
        "trade_date",
        "main_net_inflow",
        "super_large_net_inflow",
        "large_net_inflow",
        "main_net_inflow_ratio",
    ]
    CALENDAR_COLUMNS = ["trade_date", "is_open"]
    LIMIT_COLUMNS = [
        "stock_code",
        "trade_date",
        "limit_up",
        "limit_down",
        "is_limit_up",
        "is_limit_down",
        "is_suspended",
    ]

    def __init__(
        self,
        config: dict[str, Any],
        *,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.external_config = config.get("external", {})
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update({"User-Agent": self.UA})
        self.logger = logger or logging.getLogger(__name__)
        data_config = config.get("data", {})
        raw_path = resolve_path(config, data_config.get("raw_path", "data/raw"))
        self.cache_dir = raw_path / "astock_skill"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = bool(self.external_config.get("cache_enabled", True))
        self.timeout = int(self.external_config.get("request_timeout", 15))
        self.em_min_interval = float(self.external_config.get("em_min_interval", 1.0))
        self.show_progress = bool(self.external_config.get("show_progress", False))
        self.max_fetch_codes = self.external_config.get("max_fetch_codes")
        self._last_em_call = 0.0

    def fetch_stock_basic(self) -> pd.DataFrame:
        """Fetch full-market stock metadata from Eastmoney clist."""

        return self._fetch_with_cache("stock_basic", self._fetch_stock_basic_live, self.STOCK_BASIC_COLUMNS)

    def fetch_daily_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock daily K-lines from Baidu and normalize to standard daily_bars schema."""

        def fetcher() -> pd.DataFrame:
            codes = self._target_codes()
            frames = []
            for code in self._iter_codes(codes, "daily bars"):
                frame = self._fetch_baidu_kline(code, start_date, end_date, is_index=False)
                if frame.empty:
                    frame = self._fetch_tencent_kline(code, start_date, end_date, is_index=False)
                if not frame.empty:
                    frames.append(frame)
            if not frames:
                return pd.DataFrame(columns=self.DAILY_COLUMNS)
            result = pd.concat(frames, ignore_index=True)
            return result[self.DAILY_COLUMNS].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

        return self._fetch_with_cache("daily_bars", fetcher, self.DAILY_COLUMNS, start_date, end_date)

    def fetch_index_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch standard benchmark index K-lines from Baidu."""

        def fetcher() -> pd.DataFrame:
            frames = []
            index_symbols = self.external_config.get("index_symbols", self.INDEX_SYMBOLS)
            for index_code, meta in index_symbols.items():
                code = meta.get("code", self._six_digit_code(index_code)) if isinstance(meta, dict) else self._six_digit_code(index_code)
                name = meta.get("name", index_code) if isinstance(meta, dict) else index_code
                frame = self._fetch_baidu_kline(code, start_date, end_date, is_index=True)
                if frame.empty:
                    frame = self._fetch_tencent_kline(index_code, start_date, end_date, is_index=True)
                if frame.empty:
                    continue
                frame = frame.rename(columns={"stock_code": "index_code"})
                frame["index_code"] = index_code
                frame["index_name"] = name
                frames.append(frame)
            if not frames:
                return pd.DataFrame(columns=self.INDEX_COLUMNS)
            result = pd.concat(frames, ignore_index=True)
            return result[self.INDEX_COLUMNS].sort_values(["index_code", "trade_date"]).reset_index(drop=True)

        return self._fetch_with_cache("index_bars", fetcher, self.INDEX_COLUMNS, start_date, end_date)

    def fetch_sector_map(self) -> pd.DataFrame:
        """Fetch stock-to-primary-sector mapping through Eastmoney slist."""

        def fetcher() -> pd.DataFrame:
            codes = self._target_codes()
            rows = []
            for code in self._iter_codes(codes, "sector map"):
                blocks = self._fetch_eastmoney_concept_blocks(code)
                if not blocks:
                    continue
                primary = blocks[0]
                rows.append(
                    {
                        "stock_code": self._standard_code(code),
                        "sector": primary.get("name", ""),
                        "sector_code": primary.get("code", ""),
                        "sector_change_pct": self._to_float(primary.get("change_pct"), default=np.nan),
                        "lead_stock": primary.get("lead_stock", ""),
                        "concept_tags": ",".join([item.get("name", "") for item in blocks if item.get("name")]),
                    }
                )
            return pd.DataFrame(rows, columns=self.SECTOR_MAP_COLUMNS)

        return self._fetch_with_cache("sector_map", fetcher, self.SECTOR_MAP_COLUMNS)

    def fetch_sector_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch current industry board ranking from Eastmoney clist."""

        def fetcher() -> pd.DataFrame:
            params = {
                "pn": "1",
                "pz": str(self.external_config.get("sector_page_size", 100)),
                "po": "1",
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "fs": "m:90+t:2",
                "fields": "f2,f3,f4,f6,f12,f13,f14,f104,f105,f128,f136,f140,f141,f207",
            }
            response = self._em_get(self.EASTMONEY_CLIST_URL, params=params, headers={"User-Agent": self.UA})
            data = response.json()
            items = self._diff_items(data)
            rows = []
            trade_date = pd.Timestamp(end_date).normalize()
            for item in items:
                change_pct = self._to_float(item.get("f3"), default=0.0)
                rows.append(
                    {
                        "sector": item.get("f14", ""),
                        "trade_date": trade_date,
                        "close": self._to_float(item.get("f2"), default=np.nan),
                        "sector_return_1d": change_pct / 100,
                        "sector_return_3d": change_pct / 100,
                        "turnover_amount": self._to_float(item.get("f6"), default=0.0),
                        "up_count": int(self._to_float(item.get("f104"), default=0)),
                        "down_count": int(self._to_float(item.get("f105"), default=0)),
                        "leader": item.get("f140") or item.get("f128", ""),
                    }
                )
            return pd.DataFrame(rows, columns=self.SECTOR_DAILY_COLUMNS)

        return self._fetch_with_cache("sector_daily", fetcher, self.SECTOR_DAILY_COLUMNS, start_date, end_date)

    def fetch_fund_flow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily 120-day Eastmoney fund flow for configured or cached stock codes."""

        def fetcher() -> pd.DataFrame:
            frames = []
            for code in self._iter_codes(self._target_codes(), "fund flow"):
                frame = self._fetch_stock_fund_flow_120d(code)
                if not frame.empty:
                    frames.append(frame)
            if not frames:
                return pd.DataFrame(columns=self.FUND_FLOW_COLUMNS)
            result = pd.concat(frames, ignore_index=True)
            result = self._filter_dates(result, start_date, end_date)
            return result[self.FUND_FLOW_COLUMNS].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

        return self._fetch_with_cache("fund_flow", fetcher, self.FUND_FLOW_COLUMNS, start_date, end_date)

    def fetch_trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Derive trading calendar from index K-line dates."""

        def fetcher() -> pd.DataFrame:
            index_bars = self.fetch_index_bars(start_date, end_date)
            if index_bars.empty:
                daily = self.fetch_daily_bars(start_date, end_date)
                dates = daily["trade_date"] if "trade_date" in daily.columns else []
            else:
                dates = index_bars["trade_date"]
            unique_dates = sorted(pd.to_datetime(pd.Series(dates)).dropna().dt.normalize().unique())
            return pd.DataFrame({"trade_date": unique_dates, "is_open": True}, columns=self.CALENDAR_COLUMNS)

        return self._fetch_with_cache("trading_calendar", fetcher, self.CALENDAR_COLUMNS, start_date, end_date)

    def fetch_limit_status(self, date: str) -> pd.DataFrame:
        """Calculate A-share limit status from daily bars and rough board rules."""

        def fetcher() -> pd.DataFrame:
            end = pd.Timestamp(date).normalize()
            start = (end - timedelta(days=14)).date().isoformat()
            bars = self.fetch_daily_bars(start, date)
            if bars.empty:
                return pd.DataFrame(columns=self.LIMIT_COLUMNS)
            snapshot = bars[pd.to_datetime(bars["trade_date"]).dt.normalize() == end].copy()
            if snapshot.empty:
                return pd.DataFrame(columns=self.LIMIT_COLUMNS)
            stock_basic = self.fetch_stock_basic()
            if not stock_basic.empty:
                snapshot = snapshot.merge(
                    stock_basic[["stock_code", "stock_name", "is_st"]].drop_duplicates("stock_code"),
                    on="stock_code",
                    how="left",
                )
            else:
                snapshot["stock_name"] = ""
                snapshot["is_st"] = False
            return self._calculate_limit_status(snapshot)

        return self._fetch_with_cache(f"limit_status_{date}", fetcher, self.LIMIT_COLUMNS)

    def _fetch_stock_basic_live(self) -> pd.DataFrame:
        page_size = int(self.external_config.get("stock_list_page_size", 100))
        items: list[dict[str, Any]] = []
        total = None
        page = 1
        while True:
            params = {
                "pn": str(page),
                "pz": str(page_size),
                "po": "1",
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "fs": self.external_config.get(
                    "stock_list_fs",
                    "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
                ),
                "fields": "f2,f3,f8,f12,f13,f14,f17,f20,f21,f100",
            }
            response = self._em_get(self.EASTMONEY_CLIST_URL, params=params, headers={"User-Agent": self.UA})
            payload = response.json()
            data = payload.get("data") or {}
            page_items = self._diff_items(payload)
            if total is None:
                total = int(self._to_float(data.get("total"), default=len(page_items)))
            if not page_items:
                break
            items.extend(page_items)
            if len(items) >= total or len(page_items) < page_size:
                break
            page += 1
        rows = []
        for item in items:
            code = self._standard_code(str(item.get("f12", "")))
            if not code:
                continue
            name = str(item.get("f14", ""))
            list_date = pd.NaT
            rows.append(
                {
                    "stock_code": code,
                    "stock_name": name,
                    "exchange": self._exchange_for_code(code),
                    "market": item.get("f13", ""),
                    "is_st": bool("ST" in name.upper()),
                    "is_suspended": self._to_float(item.get("f2"), default=np.nan) == 0,
                    "list_date": list_date,
                    "listing_days": np.nan,
                    "sector": item.get("f100", ""),
                    "total_market_cap": self._to_float(item.get("f20"), default=np.nan),
                    "float_market_cap": self._to_float(item.get("f21"), default=np.nan),
                }
            )
        result = pd.DataFrame(rows, columns=self.STOCK_BASIC_COLUMNS)
        if not result.empty:
            result["is_st"] = result["is_st"].astype(object)
        return result

    def _fetch_baidu_kline(self, code: str, start_date: str, end_date: str, *, is_index: bool) -> pd.DataFrame:
        standard_code = self._standard_code(code)
        params = {
            "all": "1",
            "isIndex": "true" if is_index else "false",
            "isBk": "false",
            "isBlock": "false",
            "isFutures": "false",
            "isStock": "false" if is_index else "true",
            "newFormat": "1",
            "group": "quotation_kline_ab",
            "finClientType": "pc",
            "code": self._six_digit_code(code),
            "start_time": pd.Timestamp(start_date).strftime("%Y%m%d"),
            "ktype": "1",
        }
        headers = {
            "User-Agent": self.UA,
            "Accept": "application/vnd.finance-web.v1+json",
            "Origin": "https://gushitong.baidu.com",
            "Referer": "https://gushitong.baidu.com/",
        }
        response = self.session.get(self.BAIDU_KLINE_URL, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        market_data = (payload.get("Result", {}) or {}).get("newMarketData", {}) or {}
        keys = market_data.get("keys", []) or []
        row_blob = market_data.get("marketData", "") or ""
        rows = []
        for line in str(row_blob).split(";"):
            if not line.strip():
                continue
            parts = line.split(",")
            row = {keys[index]: parts[index] for index in range(min(len(keys), len(parts)))}
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=self.INDEX_COLUMNS if is_index else self.DAILY_COLUMNS)

        frame = pd.DataFrame(rows)
        date_column = "time" if "time" in frame.columns else "date"
        frame["trade_date"] = pd.to_datetime(frame[date_column].astype(str), errors="coerce").dt.normalize()
        for column in ["open", "high", "low", "close", "volume", "amount"]:
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)
        frame["prev_close"] = frame["close"].shift(1)
        if "pct_chg" in frame.columns:
            frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
        else:
            frame["pct_chg"] = (frame["close"] / frame["prev_close"] - 1) * 100
        frame["turnover_amount"] = frame["amount"]
        frame["stock_code"] = standard_code
        frame["is_suspended"] = frame["volume"].fillna(0) <= 0
        frame = self._filter_dates(frame, start_date, end_date)
        columns = self.DAILY_COLUMNS
        return frame[columns].reset_index(drop=True)

    def _fetch_tencent_kline(self, code: str, start_date: str, end_date: str, *, is_index: bool) -> pd.DataFrame:
        symbol = self._tencent_symbol(code)
        adjust = self.external_config.get("tencent_adjust", "qfq")
        kline_key = f"{adjust}day" if adjust in {"qfq", "hfq"} else "day"
        params = {
            "param": f"{symbol},day,{pd.Timestamp(start_date).date().isoformat()},"
            f"{pd.Timestamp(end_date).date().isoformat()},320,{adjust}"
        }
        response = self.session.get(
            self.TENCENT_KLINE_URL,
            params=params,
            headers={"User-Agent": self.UA, "Referer": "https://gu.qq.com/"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        node = (payload.get("data") or {}).get(symbol, {}) or {}
        rows = node.get(kline_key) or node.get("day") or node.get("qfqday") or []
        parsed = []
        for row in rows:
            if len(row) < 6:
                continue
            close = self._to_float(row[2], default=np.nan)
            volume = self._to_float(row[5], default=0.0)
            amount = self._to_float(row[6], default=np.nan) if len(row) > 6 else np.nan
            if pd.isna(amount):
                amount = close * volume * 100
            parsed.append(
                {
                    "stock_code": self._standard_code(code),
                    "trade_date": pd.to_datetime(row[0], errors="coerce").normalize(),
                    "open": self._to_float(row[1], default=np.nan),
                    "close": close,
                    "high": self._to_float(row[3], default=np.nan),
                    "low": self._to_float(row[4], default=np.nan),
                    "volume": volume,
                    "amount": amount,
                    "turnover_amount": amount,
                    "is_suspended": volume <= 0,
                }
            )
        frame = pd.DataFrame(parsed)
        if frame.empty:
            return pd.DataFrame(columns=self.DAILY_COLUMNS)
        frame = frame.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)
        frame["prev_close"] = frame["close"].shift(1)
        frame["pct_chg"] = (frame["close"] / frame["prev_close"] - 1) * 100
        frame = self._filter_dates(frame, start_date, end_date)
        return frame[self.DAILY_COLUMNS].reset_index(drop=True)

    def _fetch_eastmoney_concept_blocks(self, code: str) -> list[dict[str, Any]]:
        secid = self._secid(code)
        params = {
            "fltt": "2",
            "invt": "2",
            "secid": secid,
            "spt": "3",
            "pi": "0",
            "pz": "200",
            "po": "1",
            "fields": "f12,f14,f3,f128",
        }
        response = self._em_get(
            self.EASTMONEY_SLIST_URL,
            params=params,
            headers={"User-Agent": self.UA, "Referer": "https://quote.eastmoney.com/"},
        )
        items = self._diff_items(response.json())
        return [
            {
                "name": item.get("f14", ""),
                "code": item.get("f12", ""),
                "change_pct": item.get("f3", ""),
                "lead_stock": item.get("f128", ""),
            }
            for item in items
        ]

    def _fetch_stock_fund_flow_120d(self, code: str) -> pd.DataFrame:
        params = {
            "secid": self._secid(code),
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "lmt": str(self.external_config.get("fund_flow_limit", 120)),
        }
        response = self._em_get(
            self.EASTMONEY_FUND_FLOW_DAY_URL,
            params=params,
            headers={
                "User-Agent": self.UA,
                "Referer": "https://quote.eastmoney.com/",
                "Origin": "https://quote.eastmoney.com",
            },
        )
        klines = (response.json().get("data", {}) or {}).get("klines", []) or []
        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "stock_code": self._standard_code(code),
                    "trade_date": pd.to_datetime(parts[0], errors="coerce").normalize(),
                    "main_net_inflow": self._to_float(parts[1]),
                    "super_large_net_inflow": self._to_float(parts[5]),
                    "large_net_inflow": self._to_float(parts[4]),
                    "main_net_inflow_ratio": 0.0,
                }
            )
        return pd.DataFrame(rows, columns=self.FUND_FLOW_COLUMNS)

    def _calculate_limit_status(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        frame = snapshot.copy()
        frame["prev_close"] = pd.to_numeric(frame["prev_close"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")

        def pct_limit(row: pd.Series) -> float:
            code = self._six_digit_code(row["stock_code"])
            name = str(row.get("stock_name", ""))
            if bool(row.get("is_st", False)) or "ST" in name.upper():
                return 0.05
            if code.startswith(("300", "301", "688", "689")):
                return 0.20
            if code.startswith(("4", "8")):
                return 0.30
            return 0.10

        limits = frame.apply(pct_limit, axis=1)
        frame["limit_up"] = (frame["prev_close"] * (1 + limits)).round(2)
        frame["limit_down"] = (frame["prev_close"] * (1 - limits)).round(2)
        frame["is_limit_up"] = frame["close"] >= frame["limit_up"] * 0.999
        frame["is_limit_down"] = frame["close"] <= frame["limit_down"] * 1.001
        if "is_suspended" not in frame.columns:
            frame["is_suspended"] = False
        return frame[self.LIMIT_COLUMNS].reset_index(drop=True)

    def _fetch_with_cache(
        self,
        name: str,
        fetcher: Callable[[], pd.DataFrame],
        columns: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        try:
            result = fetcher()
            result = self._ensure_columns(result, columns)
            if start_date and end_date and "trade_date" in result.columns:
                result = self._filter_dates(result, start_date, end_date)
            if not result.empty:
                self._write_cache(name, result)
                return result.reset_index(drop=True)
            cached = self._read_cache(name, columns)
            if not cached.empty:
                if start_date and end_date and "trade_date" in cached.columns:
                    cached = self._filter_dates(cached, start_date, end_date)
                self.logger.warning("live source returned empty for %s; using cached data", name)
                return cached.reset_index(drop=True)
            return pd.DataFrame(columns=columns)
        except Exception as exc:
            self.logger.warning("live source failed for %s: %s", name, exc)
            cached = self._read_cache(name, columns)
            if not cached.empty:
                if start_date and end_date and "trade_date" in cached.columns:
                    cached = self._filter_dates(cached, start_date, end_date)
                return cached.reset_index(drop=True)
            return pd.DataFrame(columns=columns)

    def _em_get(self, url: str, params: dict[str, Any], headers: dict[str, str] | None = None):
        wait = self.em_min_interval - (time.time() - self._last_em_call)
        if wait > 0:
            time.sleep(wait + random.uniform(0.01, 0.05))
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response
        finally:
            self._last_em_call = time.time()

    def _target_codes(self) -> list[str]:
        configured = self.external_config.get("astock_codes") or self.external_config.get("stock_codes") or []
        if configured:
            return self._limit_codes([self._standard_code(code) for code in configured])
        cached_basic = self._read_cache("stock_basic", self.STOCK_BASIC_COLUMNS)
        if cached_basic.empty:
            cached_basic = self.fetch_stock_basic()
        if cached_basic.empty or "stock_code" not in cached_basic.columns:
            return []
        return self._limit_codes(cached_basic["stock_code"].dropna().astype(str).tolist())

    def _limit_codes(self, codes: list[str]) -> list[str]:
        cleaned = [code for code in dict.fromkeys(codes) if code]
        if self.max_fetch_codes in (None, "", 0, "0"):
            return cleaned
        return cleaned[: int(self.max_fetch_codes)]

    def _iter_codes(self, codes: list[str], label: str):
        iterator: Iterable[str] = codes
        if self.show_progress:
            iterator = tqdm(codes, desc=f"fetch {label}")
        return iterator

    def _write_cache(self, name: str, df: pd.DataFrame) -> None:
        if not self.cache_enabled or df.empty:
            return
        path = self.cache_dir / f"{name}.parquet"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        except Exception as exc:
            self.logger.warning("failed writing source cache %s: %s", path, exc)

    def _read_cache(self, name: str, columns: list[str]) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.parquet"
        if not self.cache_enabled or not path.exists():
            return pd.DataFrame(columns=columns)
        try:
            return self._ensure_columns(pd.read_parquet(path), columns)
        except Exception as exc:
            self.logger.warning("failed reading source cache %s: %s", path, exc)
            return pd.DataFrame(columns=columns)

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            if column not in result.columns:
                result[column] = pd.NA
        return result[columns]

    @staticmethod
    def _filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df.empty or "trade_date" not in df.columns:
            return df
        result = df.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        return result[(result["trade_date"] >= start) & (result["trade_date"] <= end)].reset_index(drop=True)

    @staticmethod
    def _diff_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        diff = (payload.get("data") or {}).get("diff") or []
        if isinstance(diff, dict):
            return list(diff.values())
        return list(diff)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value in (None, "", "-", "--"):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _standard_code(cls, code: str) -> str:
        six = cls._six_digit_code(code)
        if not six:
            return ""
        return f"{six}.{cls._exchange_for_code(six)}"

    @staticmethod
    def _six_digit_code(code: str) -> str:
        value = str(code).strip().upper()
        if "." in value:
            value = value.split(".")[0]
        if value.startswith(("SH", "SZ", "BJ")):
            value = value[2:]
        digits = "".join(ch for ch in value if ch.isdigit())
        return digits[-6:] if len(digits) >= 6 else digits

    @classmethod
    def _exchange_for_code(cls, code: str) -> str:
        six = cls._six_digit_code(code)
        if six.startswith(("6", "9")):
            return "SH"
        if six.startswith(("4", "8")):
            return "BJ"
        return "SZ"

    @classmethod
    def _secid(cls, code: str) -> str:
        six = cls._six_digit_code(code)
        market_code = 1 if six.startswith(("6", "9")) else 0
        return f"{market_code}.{six}"

    @classmethod
    def _tencent_symbol(cls, code: str) -> str:
        six = cls._six_digit_code(code)
        exchange = cls._exchange_for_code(code).lower()
        return f"{exchange}{six}"
