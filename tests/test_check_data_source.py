from scripts.check_data_source import probe_baidu_daily


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_probe_baidu_daily_alive_with_sample_fields():
    def fake_get(url, params, headers, timeout):
        return _Response(200, {"ResultCode": "0", "Result": {"newMarketData": [{"date": "20260604", "close": "10"}]}})

    result = probe_baidu_daily(request_get=fake_get, retries=1)

    assert result["status"] == "ALIVE"
    assert result["status_code"] == 200
    assert "newMarketData" in result["sample_fields"]


def test_probe_baidu_daily_dead_on_exception():
    def fake_get(url, params, headers, timeout):
        raise TimeoutError("network unavailable")

    result = probe_baidu_daily(request_get=fake_get, retries=1)

    assert result["status"] == "DEAD"
    assert "network unavailable" in result["error"]
