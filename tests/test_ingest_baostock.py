import pandas as pd

import scripts.ingest_baostock as ingest
from scripts.ingest_baostock import (
    Throttle,
    _ingest_daily_bars_incremental,
    _enumerate_stock_codes,
    _fetch_sector_map,
    _fetch_stock_basic,
    _login_with_retry,
    _looks_like_connection_error,
    _query_result,
    baostock_to_standard_code,
    build_arg_parser,
    build_sector_daily,
    derive_limit_flags,
    normalize_daily_bars,
    normalize_stock_basic,
    standard_to_baostock_code,
)
from astock_quant.data.storage import StorageManager


class _RS:
    """Minimal baostock result-set stand-in."""

    def __init__(self, rows, fields, error_code="0", error_msg="success"):
        self._rows = list(rows)
        self.fields = list(fields)
        self.error_code = error_code
        self.error_msg = error_msg
        self._index = -1

    def next(self):
        self._index += 1
        return self._index < len(self._rows)

    def get_row_data(self):
        return self._rows[self._index]


class _FakeLogin:
    error_code = "0"
    error_msg = "success"


def test_baostock_code_conversion_round_trips():
    assert baostock_to_standard_code("sh.600000") == "600000.SH"
    assert baostock_to_standard_code("sz.000001") == "000001.SZ"
    assert standard_to_baostock_code("600000.SH") == "sh.600000"
    assert standard_to_baostock_code("000001.SZ") == "sz.000001"


def test_normalize_daily_bars_maps_qfq_standard_fields_and_decimal_pct_chg():
    raw = pd.DataFrame(
        [
            {
                "date": "2026-06-04",
                "code": "sh.600000",
                "open": "10.00",
                "high": "10.90",
                "low": "9.90",
                "close": "10.99",
                "preclose": "10.00",
                "volume": "1000",
                "amount": "12345678.9",
                "turn": "1.23",
                "tradestatus": "1",
                "pctChg": "9.90",
                "isST": "0",
            },
            {
                "date": "2026-06-04",
                "code": "sz.300001",
                "open": "20.00",
                "high": "20.10",
                "low": "16.00",
                "close": "16.02",
                "preclose": "20.00",
                "volume": "0",
                "amount": "0",
                "turn": "",
                "tradestatus": "0",
                "pctChg": "-19.90",
                "isST": "0",
            },
        ]
    )

    result = normalize_daily_bars(raw)

    row = result.set_index("stock_code").loc["600000.SH"]
    assert row["trade_date"] == "2026-06-04"
    assert row["prev_close"] == 10.0
    assert row["turnover_amount"] == 12345678.9
    assert row["turnover_rate"] == 1.23
    assert row["pct_chg"] == 0.099
    assert row["adjust_type"] == "qfq"
    assert bool(row["is_limit_up"]) is True
    assert bool(row["is_suspended"]) is False

    suspended = result.set_index("stock_code").loc["300001.SZ"]
    assert bool(suspended["is_suspended"]) is True
    assert bool(suspended["is_limit_down"]) is True


def test_derive_limit_flags_uses_st_and_board_specific_limits():
    frame = pd.DataFrame(
        [
            {"stock_code": "600001.SH", "pct_chg": 0.0491, "is_st": True},
            {"stock_code": "688001.SH", "pct_chg": 0.1991, "is_st": False},
            {"stock_code": "000001.SZ", "pct_chg": -0.0991, "is_st": False},
        ]
    )

    result = derive_limit_flags(frame).set_index("stock_code")

    assert bool(result.loc["600001.SH", "is_limit_up"]) is True
    assert bool(result.loc["688001.SH", "is_limit_up"]) is True
    assert bool(result.loc["000001.SZ", "is_limit_down"]) is True


def test_normalize_stock_basic_does_not_fabricate_float_market_cap():
    raw = pd.DataFrame(
        [
            {
                "code": "sh.600000",
                "code_name": "浦发银行",
                "ipoDate": "1999-11-10",
                "outDate": "",
                "status": "1",
                "type": "1",
            },
            {
                "code": "sz.000001",
                "code_name": "平安银行",
                "ipoDate": "1991-04-03",
                "outDate": "2026-01-01",
                "status": "0",
                "type": "1",
            },
        ]
    )

    result = normalize_stock_basic(raw)

    assert "float_market_cap" not in result.columns
    by_code = result.set_index("stock_code")
    assert by_code.loc["600000.SH", "exchange"] == "SH"
    assert bool(by_code.loc["000001.SZ", "is_delisted"]) is True
    assert by_code.loc["600000.SH", "is_st"] is None


def test_build_sector_daily_aggregates_daily_bars_by_industry():
    daily_bars = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "trade_date": "2026-06-04",
                "close": 10.0,
                "turnover_amount": 100.0,
                "is_limit_up": True,
            },
            {
                "stock_code": "600002.SH",
                "trade_date": "2026-06-04",
                "close": 20.0,
                "turnover_amount": 300.0,
                "is_limit_up": False,
            },
        ]
    )
    sector_map = pd.DataFrame(
        [
            {
                "stock_code": "600001.SH",
                "sector_code": "银行",
                "sector_name": "银行",
                "sector_type": "industry",
            },
            {
                "stock_code": "600002.SH",
                "sector_code": "银行",
                "sector_name": "银行",
                "sector_type": "industry",
            },
        ]
    )

    result = build_sector_daily(daily_bars, sector_map)

    row = result.iloc[0]
    assert row["sector_code"] == "银行"
    assert row["sector_stock_count"] == 2
    assert row["limit_up_count"] == 1
    assert row["amount"] == 400.0
    assert row["turnover_amount"] == 400.0
    assert row["close"] == 17.5


def test_login_with_retry_retries_until_success():
    class FakeResult:
        def __init__(self, error_code: str, error_msg: str = ""):
            self.error_code = error_code
            self.error_msg = error_msg

    class FakeBaostock:
        def __init__(self):
            self.calls = 0

        def login(self):
            self.calls += 1
            if self.calls == 1:
                return FakeResult("10002007", "network error")
            return FakeResult("0", "")

    fake = FakeBaostock()

    result = _login_with_retry(fake, retries=2, sleep=lambda _: None)

    assert result.error_code == "0"
    assert fake.calls == 2


class FakeResultSet:
    def __init__(self, fields, rows, error_code: str = "0", error_msg: str = ""):
        self.fields = fields
        self.rows = rows
        self.error_code = error_code
        self.error_msg = error_msg
        self.index = -1

    def next(self):
        self.index += 1
        return self.index < len(self.rows)

    def get_row_data(self):
        return self.rows[self.index]


def test_fetch_stock_basic_with_codes_never_calls_global_query_stock_basic():
    class FakeBaostock:
        def __init__(self):
            self.basic_calls = []

        def query_stock_basic(self, code=None):
            self.basic_calls.append(code)
            if code is None:
                raise AssertionError("global query_stock_basic must not be called")
            return FakeResultSet(
                ["code", "code_name", "ipoDate", "outDate", "type", "status"],
                [[code, f"name-{code}", "2000-01-01", "", "1", "1"]],
            )

    fake = FakeBaostock()

    result = _fetch_stock_basic(fake, True, ["sh.600000", "sz.000001"], None)

    assert fake.basic_calls == ["sh.600000", "sz.000001"]
    assert result["code"].tolist() == ["sh.600000", "sz.000001"]


def test_fetch_stock_basic_without_codes_enumerates_query_all_stock_first():
    class FakeBaostock:
        def __init__(self):
            self.all_stock_calls = []
            self.basic_calls = []

        def query_all_stock(self, day=None):
            self.all_stock_calls.append(day)
            return FakeResultSet(["code", "tradeStatus"], [["sh.600000", "1"], ["sz.000001", "1"]])

        def query_stock_basic(self, code=None):
            self.basic_calls.append(code)
            if code is None:
                raise AssertionError("global query_stock_basic must not be called")
            return FakeResultSet(
                ["code", "code_name", "ipoDate", "outDate", "type", "status"],
                [[code, f"name-{code}", "2000-01-01", "", "1", "1"]],
            )

    fake = FakeBaostock()

    result = _fetch_stock_basic(fake, True, [], None, start_date="2024-01-01", end_date="2024-06-30")

    assert fake.all_stock_calls == ["2024-06-30"]
    assert fake.basic_calls == ["sh.600000", "sz.000001"]
    assert result["code"].tolist() == ["sh.600000", "sz.000001"]


def test_enumerate_stock_codes_excludes_index_codes():
    class FakeBaostock:
        def query_all_stock(self, day=None):
            return FakeResultSet(
                ["code", "tradeStatus"],
                [
                    ["sh.000001", "1"],
                    ["sh.000300", "1"],
                    ["sz.399006", "1"],
                    ["bj.430001", "1"],
                    ["sh.600000", "1"],
                    ["sh.601001", "1"],
                    ["sh.603001", "1"],
                    ["sh.605001", "1"],
                    ["sh.688001", "1"],
                    ["sz.000001", "1"],
                    ["sz.001001", "1"],
                    ["sz.002001", "1"],
                    ["sz.003001", "1"],
                    ["sz.300001", "1"],
                    ["sz.301001", "1"],
                ],
            )

    codes = _enumerate_stock_codes(FakeBaostock(), None, "2024-06-30", include_delisted=False)

    assert codes == [
        "sh.600000",
        "sh.601001",
        "sh.603001",
        "sh.605001",
        "sh.688001",
        "sz.000001",
        "sz.001001",
        "sz.002001",
        "sz.003001",
        "sz.300001",
        "sz.301001",
    ]


def test_fetch_stock_basic_applies_limit_after_true_stock_filtering():
    class FakeBaostock:
        def __init__(self):
            self.basic_calls = []

        def query_all_stock(self, day=None):
            return FakeResultSet(
                ["code", "tradeStatus"],
                [
                    ["sh.000001", "1"],
                    ["sh.000300", "1"],
                    ["sz.399006", "1"],
                    ["sh.600000", "1"],
                    ["sz.000001", "1"],
                    ["sh.688001", "1"],
                ],
            )

        def query_stock_basic(self, code=None):
            self.basic_calls.append(code)
            return FakeResultSet(
                ["code", "code_name", "ipoDate", "outDate", "type", "status"],
                [[code, f"name-{code}", "2000-01-01", "", "1", "1"]],
            )

    fake = FakeBaostock()

    result = _fetch_stock_basic(fake, True, [], 2, start_date="2024-01-01", end_date="2024-06-30")

    assert fake.basic_calls == ["sh.600000", "sz.000001"]
    assert result["code"].tolist() == ["sh.600000", "sz.000001"]


def test_fetch_stock_basic_skips_failed_code_and_keeps_others():
    class FakeBaostock:
        def query_stock_basic(self, code=None):
            if code == "sh.600001":
                raise TimeoutError("socket timed out")
            return FakeResultSet(
                ["code", "code_name", "ipoDate", "outDate", "type", "status"],
                [[code, f"name-{code}", "2000-01-01", "", "1", "1"]],
            )

    failed: list[str] = []

    result = _fetch_stock_basic(
        FakeBaostock(),
        True,
        ["sh.600000", "sh.600001", "sz.000001"],
        None,
        failed_codes=failed,
        retries=1,
    )

    assert failed == ["sh.600001"]
    assert result["code"].tolist() == ["sh.600000", "sz.000001"]


def test_fetch_sector_map_queries_industry_per_code_and_fills_unknown():
    class FakeBaostock:
        def __init__(self):
            self.industry_calls = []

        def query_stock_industry(self, code=None):
            self.industry_calls.append(code)
            if code is None:
                raise AssertionError("global query_stock_industry must not be called")
            if code == "sh.600000":
                return FakeResultSet(["code", "industry"], [[code, "银行"]])
            return FakeResultSet(["code", "industry"], [])

    fake = FakeBaostock()

    result = _fetch_sector_map(fake, ["sh.600000", "sz.000001"])

    assert fake.industry_calls == ["sh.600000", "sz.000001"]
    assert result.set_index("code").loc["sz.000001", "industry"] == "unknown"


def test_fetch_sector_map_records_failed_code_as_unknown():
    class FakeBaostock:
        def query_stock_industry(self, code=None):
            if code == "sz.000001":
                raise TimeoutError("socket timed out")
            return FakeResultSet(["code", "industry"], [[code, "银行"]])

    failed: list[str] = []

    result = _fetch_sector_map(FakeBaostock(), ["sh.600000", "sz.000001"], failed_codes=failed, retries=1)

    assert failed == ["sz.000001"]
    assert result.set_index("code").loc["sz.000001", "industry"] == "unknown"


def test_build_arg_parser_has_timeout_and_save_every_defaults():
    args = build_arg_parser().parse_args(["--start", "2024-01-01", "--end", "2024-06-30"])

    assert args.timeout == 30
    assert args.save_every == 50
    assert args.sleep == 0.0
    assert args.batch_size == 0
    assert args.batch_rest == 0.0
    assert args.jitter == 0.2
    assert args.relogin_every == 0


def test_build_arg_parser_parses_throttle_flags():
    args = build_arg_parser().parse_args(
        ["--start", "2024-01-01", "--end", "2024-06-30", "--sleep", "0.5", "--batch-size", "200", "--batch-rest", "60"]
    )

    assert args.sleep == 0.5
    assert args.batch_size == 200
    assert args.batch_rest == 60.0


def test_throttle_default_is_noop():
    slept = []
    throttle = Throttle(_sleep=slept.append)

    for _ in range(5):
        throttle.tick()

    assert slept == []


def test_throttle_sleeps_between_requests_without_jitter():
    slept = []
    throttle = Throttle(sleep_seconds=0.5, jitter=0.0, _sleep=slept.append)

    for _ in range(3):
        throttle.tick()

    assert slept == [0.5, 0.5, 0.5]


def test_throttle_takes_long_rest_every_batch():
    slept = []
    throttle = Throttle(sleep_seconds=0.5, batch_size=3, batch_rest_seconds=60.0, jitter=0.0, _sleep=slept.append)

    for _ in range(6):
        throttle.tick()

    # Short pause on requests 1,2,4,5; long rest on requests 3 and 6.
    assert slept == [0.5, 0.5, 60.0, 0.5, 0.5, 60.0]


def test_throttle_jitter_stays_within_bounds():
    slept = []
    throttle = Throttle(sleep_seconds=1.0, jitter=0.2, _sleep=slept.append, _rand=lambda lo, hi: hi)

    throttle.tick()

    assert slept == [1.2]


def test_looks_like_connection_error_matches_baostock_receive_errors():
    assert _looks_like_connection_error("daily sz.002491 failed: 10002007 网络接收错误")
    assert _looks_like_connection_error("[WinError 10054] 远程主机强迫关闭了一个现有的连接")
    assert _looks_like_connection_error("ConnectionResetError: connection reset by peer")
    # A genuine business error (bad params, not a dropped socket) must not trigger relogin.
    assert not _looks_like_connection_error("query failed: 10004003 invalid date range")


def test_query_result_relogins_on_connection_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda *_: None)

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0
            self.logout_calls = 0

        def login(self):
            self.login_calls += 1
            return _FakeLogin()

        def logout(self):
            self.logout_calls += 1

    bs = FakeBaostock()
    attempts = {"n": 0}

    def factory(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _RS([], ["code"], error_code="10002007", error_msg="网络接收错误")
        return _RS([["600000.SH"]], ["code"])

    result = _query_result(factory, "daily test", retries=3, bs=bs, code="sh.600000")

    assert attempts["n"] == 2  # failed once, retried after relogin, then succeeded
    assert bs.logout_calls == 1
    assert bs.login_calls == 1
    assert result["code"].tolist() == ["600000.SH"]


def test_query_result_does_not_relogin_on_business_error(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda *_: None)

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0

        def login(self):
            self.login_calls += 1
            return _FakeLogin()

        def logout(self):
            pass

    bs = FakeBaostock()

    def factory(**kwargs):
        return _RS([], ["code"], error_code="10004003", error_msg="invalid params")

    try:
        _query_result(factory, "daily test", retries=2, bs=bs, code="sh.600000")
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError after retries")

    assert bs.login_calls == 0  # business errors must not trigger a re-login


def test_proactive_relogin_fires_on_interval(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda *_: None)
    relogins = []
    monkeypatch.setattr(ingest, "_relogin", lambda bs, **kw: relogins.append(True))

    for processed in range(1, 7):
        ingest._proactive_relogin(object(), processed, relogin_every=3)

    assert len(relogins) == 2  # fired at 3 and 6


def test_fetch_stock_basic_ticks_throttle_per_code():
    class FakeResultSet:
        def __init__(self):
            self._done = False

        @property
        def error_code(self):
            return "0"

        fields = ["code", "code_name", "ipoDate", "outDate", "type", "status"]

        def next(self):
            if self._done:
                return False
            self._done = True
            return True

        def get_row_data(self):
            return ["sh.600000", "x", "2000-01-01", "", "1", "1"]

    class FakeBaostock:
        def query_stock_basic(self, code):
            return FakeResultSet()

    slept = []
    throttle = Throttle(sleep_seconds=0.3, jitter=0.0, _sleep=slept.append)
    _fetch_stock_basic(FakeBaostock(), True, ["sh.600000", "sz.000001"], None, throttle=throttle)

    assert slept == [0.3, 0.3]


def test_daily_ingest_skips_failed_code_and_incrementally_saves(tmp_path):
    class FakeBaostock:
        def __init__(self):
            self.daily_calls = []

        def query_history_k_data_plus(self, code, fields, **kwargs):
            self.daily_calls.append(code)
            if code == "sh.600001":
                raise TimeoutError("socket timed out")
            return FakeResultSet(
                [
                    "date",
                    "code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "preclose",
                    "volume",
                    "amount",
                    "turn",
                    "tradestatus",
                    "pctChg",
                    "isST",
                ],
                [[kwargs["start_date"], code, "10", "10.5", "9.8", "10.2", "10", "100", "1000", "1", "1", "2", "0"]],
            )

    storage = _tmp_storage(tmp_path)
    failed: list[str] = []

    result = _ingest_daily_bars_incremental(
        FakeBaostock(),
        storage,
        ["sh.600000", "sh.600001", "sz.000001"],
        pd.DataFrame(),
        "2024-01-01",
        "2024-01-02",
        save_every=1,
        failed_codes=failed,
        retries=1,
    )

    saved = storage.read_parquet("daily_bars.parquet")
    assert failed == ["sh.600001"]
    assert result["stock_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert saved["stock_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert saved["trade_date"].map(type).eq(str).all()


def test_daily_ingest_rerun_skips_existing_covered_code(tmp_path):
    class FakeBaostock:
        def __init__(self):
            self.daily_calls = []

        def query_history_k_data_plus(self, code, fields, **kwargs):
            self.daily_calls.append(code)
            return FakeResultSet(
                [
                    "date",
                    "code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "preclose",
                    "volume",
                    "amount",
                    "turn",
                    "tradestatus",
                    "pctChg",
                    "isST",
                ],
                [[kwargs["start_date"], code, "20", "21", "19", "20.5", "20", "200", "2000", "2", "1", "2.5", "0"]],
            )

    storage = _tmp_storage(tmp_path)
    existing = pd.DataFrame(
        [
            {
                "stock_code": "600000.SH",
                "trade_date": pd.Timestamp("2024-01-01"),
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "prev_close": 10,
                "volume": 100,
                "turnover_amount": 1000,
                "turnover_rate": 1,
                "is_suspended": False,
                "is_st": False,
                "pct_chg": 0,
                "adjust_type": "qfq",
                "is_limit_up": False,
                "is_limit_down": False,
            },
            {
                "stock_code": "600000.SH",
                "trade_date": pd.Timestamp("2024-01-02"),
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "prev_close": 10,
                "volume": 100,
                "turnover_amount": 1000,
                "turnover_rate": 1,
                "is_suspended": False,
                "is_st": False,
                "pct_chg": 0,
                "adjust_type": "qfq",
                "is_limit_up": False,
                "is_limit_down": False,
            },
        ]
    )
    storage.save_parquet(existing, "daily_bars.parquet")
    fake = FakeBaostock()

    result = _ingest_daily_bars_incremental(
        fake,
        storage,
        ["sh.600000", "sz.000001"],
        storage.read_parquet("daily_bars.parquet"),
        "2024-01-01",
        "2024-01-02",
        save_every=1,
        failed_codes=[],
        retries=1,
    )

    assert fake.daily_calls == ["sz.000001"]
    assert set(result["stock_code"]) == {"600000.SH", "000001.SZ"}
    assert result["trade_date"].map(type).eq(str).all()


def _tmp_storage(tmp_path) -> StorageManager:
    return StorageManager(
        {
            "data": {
                "raw_path": tmp_path / "raw",
                "processed_path": tmp_path / "processed",
                "result_path": tmp_path / "results",
                "report_path": tmp_path / "reports",
            }
        }
    )
