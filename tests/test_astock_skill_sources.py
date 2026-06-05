import pandas as pd

from astock_quant.data.astock_skill_sources import AStockSkillSource


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "params": params or {}})
        if "selfselect/getstockquotation" in url:
            return FakeResponse(
                {
                    "Result": {
                        "newMarketData": {
                            "keys": ["time", "open", "close", "high", "low", "volume", "amount"],
                            "marketData": (
                                "20260603,10,10.5,10.6,9.9,1000,100000;"
                                "20260604,10.5,11,11.2,10.4,2000,220000"
                            ),
                        }
                    }
                }
            )
        if "push2.eastmoney.com/api/qt/clist/get" in url:
            fs = (params or {}).get("fs")
            if fs == "m:90+t:2":
                return FakeResponse(
                    {
                        "data": {
                            "diff": [
                                {
                                    "f12": "BK0001",
                                    "f14": "机器人",
                                    "f2": 1000,
                                    "f3": 2.5,
                                    "f6": 5000000000,
                                    "f104": 80,
                                    "f105": 20,
                                    "f140": "强势股份",
                                }
                            ]
                        }
                    }
                )
            return FakeResponse(
                {
                    "data": {
                        "diff": [
                            {
                                "f12": "600001",
                                "f14": "强势股份",
                                "f13": 1,
                                "f100": "机器人",
                                "f20": 10000000000,
                                "f21": 5000000000,
                            },
                            {
                                "f12": "300001",
                                "f14": "ST风险",
                                "f13": 0,
                                "f100": "软件服务",
                                "f20": 8000000000,
                                "f21": 3000000000,
                            },
                        ]
                    }
                }
            )
        if "push2.eastmoney.com/api/qt/slist/get" in url:
            return FakeResponse(
                {
                    "data": {
                        "diff": [
                            {"f12": "BK0001", "f14": "机器人", "f3": 2.5, "f128": "强势股份"},
                            {"f12": "BK0002", "f14": "人工智能", "f3": 1.8, "f128": "科技股份"},
                        ]
                    }
                }
            )
        if "push2his.eastmoney.com/api/qt/stock/fflow/daykline/get" in url:
            return FakeResponse(
                {
                    "data": {
                        "klines": [
                            "2026-06-03,100,10,20,30,40",
                            "2026-06-04,200,11,21,31,41",
                        ]
                    }
                }
            )
        raise AssertionError(f"unexpected URL: {url}")


class FakeTencentFallbackSession(FakeSession):
    def get(self, url, params=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "params": params or {}})
        if "selfselect/getstockquotation" in url:
            return FakeResponse({"QueryID": "0", "ResultCode": "403", "Result": []})
        if "appstock/app/fqkline/get" in url:
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "sh600519": {
                            "qfqday": [
                                ["2024-01-02", "1608.685", "1578.695", "1611.875", "1571.785", "32156.000"],
                                ["2024-01-03", "1574.795", "1587.685", "1588.905", "1570.015", "20229.000"],
                            ]
                        }
                    },
                }
            )
        return super().get(url, params=params, headers=headers, timeout=timeout, **kwargs)


class FakePagedStockListSession(FakeSession):
    def get(self, url, params=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "params": params or {}})
        if "push2.eastmoney.com/api/qt/clist/get" in url:
            page = str((params or {}).get("pn"))
            data_by_page = {
                "1": [
                    {"f12": "600001", "f14": "强势股份", "f13": 1, "f100": "机器人", "f20": 1, "f21": 1},
                    {"f12": "600002", "f14": "科技股份", "f13": 1, "f100": "软件", "f20": 1, "f21": 1},
                ],
                "2": [
                    {"f12": "000001", "f14": "平安银行", "f13": 0, "f100": "银行", "f20": 1, "f21": 1},
                ],
            }
            return FakeResponse({"data": {"total": 3, "diff": data_by_page.get(page, [])}})
        return super().get(url, params=params, headers=headers, timeout=timeout, **kwargs)


def test_skill_source_fetches_baidu_daily_bars_in_standard_schema(tmp_path):
    source = AStockSkillSource(
        {"_project_root": str(tmp_path), "external": {"astock_codes": ["600001.SH"], "em_min_interval": 0}},
        session=FakeSession(),
    )

    bars = source.fetch_daily_bars("2026-06-03", "2026-06-04")

    assert list(bars["stock_code"].unique()) == ["600001.SH"]
    assert list(bars["trade_date"].dt.strftime("%Y-%m-%d")) == ["2026-06-03", "2026-06-04"]
    assert bars.iloc[1]["open"] == 10.5
    assert bars.iloc[1]["high"] == 11.2
    assert bars.iloc[1]["low"] == 10.4
    assert bars.iloc[1]["close"] == 11.0
    assert bars.iloc[1]["volume"] == 2000
    assert bars.iloc[1]["turnover_amount"] == 220000
    assert round(bars.iloc[1]["pct_chg"], 4) == 4.7619


def test_skill_source_falls_back_to_tencent_kline_when_baidu_returns_empty(tmp_path):
    source = AStockSkillSource(
        {"_project_root": str(tmp_path), "external": {"astock_codes": ["600519.SH"], "em_min_interval": 0}},
        session=FakeTencentFallbackSession(),
    )

    bars = source.fetch_daily_bars("2024-01-02", "2024-01-03")

    assert bars["stock_code"].tolist() == ["600519.SH", "600519.SH"]
    assert bars.iloc[0]["open"] == 1608.685
    assert bars.iloc[0]["close"] == 1578.695
    assert bars.iloc[0]["volume"] == 32156
    assert bars.iloc[0]["turnover_amount"] > 0


def test_skill_source_fetches_stock_basic_sector_and_fund_flow(tmp_path):
    source = AStockSkillSource(
        {"_project_root": str(tmp_path), "external": {"astock_codes": ["600001.SH"], "em_min_interval": 0}},
        session=FakeSession(),
    )

    stock_basic = source.fetch_stock_basic()
    sector_map = source.fetch_sector_map()
    sector_daily = source.fetch_sector_daily("2026-06-04", "2026-06-04")
    fund_flow = source.fetch_fund_flow("2026-06-04", "2026-06-04")

    assert set(["stock_code", "stock_name", "exchange", "is_st", "float_market_cap", "sector"]).issubset(
        stock_basic.columns
    )
    assert stock_basic.loc[stock_basic["stock_code"] == "300001.SZ", "is_st"].iloc[0] is True
    assert sector_map.iloc[0]["sector"] == "机器人"
    assert sector_daily.iloc[0]["sector_return_1d"] == 0.025
    assert sector_daily.iloc[0]["turnover_amount"] == 5000000000
    assert fund_flow.iloc[0]["main_net_inflow"] == 200
    assert fund_flow.iloc[0]["large_net_inflow"] == 31
    assert fund_flow.iloc[0]["super_large_net_inflow"] == 41


def test_skill_source_paginates_stock_basic_until_total_is_reached(tmp_path):
    session = FakePagedStockListSession()
    source = AStockSkillSource(
        {
            "_project_root": str(tmp_path),
            "external": {"stock_list_page_size": 2, "em_min_interval": 0},
        },
        session=session,
    )

    stock_basic = source.fetch_stock_basic()

    assert stock_basic["stock_code"].tolist() == ["600001.SH", "600002.SH", "000001.SZ"]
    stock_list_calls = [
        call for call in session.calls if "push2.eastmoney.com/api/qt/clist/get" in call["url"]
    ]
    assert [call["params"]["pn"] for call in stock_list_calls] == ["1", "2"]


def test_skill_source_calculates_limit_status_fallback(tmp_path):
    source = AStockSkillSource(
        {"_project_root": str(tmp_path), "external": {"astock_codes": ["600001.SH"], "em_min_interval": 0}},
        session=FakeSession(),
    )

    status = source.fetch_limit_status("2026-06-04")

    row = status.iloc[0]
    assert row["stock_code"] == "600001.SH"
    assert row["limit_up"] == 11.55
    assert row["limit_down"] == 9.45
    assert bool(row["is_limit_up"]) is False
    assert bool(row["is_limit_down"]) is False
