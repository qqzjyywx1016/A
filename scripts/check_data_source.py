#!/usr/bin/env python3
"""Probe whether the a-stock-data Baidu daily K-line endpoint is reachable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astock_quant.data.astock_skill_sources import AStockSkillSource


def probe_baidu_daily(
    *,
    code: str = "600519",
    timeout: float = 5.0,
    retries: int = 2,
    request_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Return ALIVE/DEAD plus response status and sample payload fields."""

    get = request_get or requests.get
    params = {
        "all": "1",
        "isIndex": "false",
        "isBk": "false",
        "isBlock": "false",
        "isFutures": "false",
        "isStock": "true",
        "newFormat": "1",
        "group": "quotation_kline_ab",
        "finClientType": "pc",
        "code": code.split(".")[0],
        "ktype": "1",
    }
    headers = {
        "User-Agent": AStockSkillSource.UA,
        "Accept": "application/vnd.finance-web.v1+json",
        "Origin": "https://gushitong.baidu.com",
        "Referer": "https://gushitong.baidu.com/",
    }
    last_error = ""
    status_code: int | None = None
    for _ in range(max(int(retries), 1)):
        try:
            response = get(AStockSkillSource.BAIDU_KLINE_URL, params=params, headers=headers, timeout=timeout)
            status_code = int(getattr(response, "status_code", 0))
            payload = response.json()
            result = payload.get("Result", {}) if isinstance(payload, dict) else {}
            code_value = str(payload.get("ResultCode", "")) if isinstance(payload, dict) else ""
            sample_fields = sorted(result.keys()) if isinstance(result, dict) else []
            if status_code == 200 and code_value in {"0", "None", ""} and sample_fields:
                return {
                    "status": "ALIVE",
                    "status_code": status_code,
                    "sample_fields": sample_fields,
                    "error": "",
                }
            last_error = f"unexpected payload ResultCode={code_value} fields={sample_fields}"
        except Exception as exc:  # pragma: no cover - exercised through tests with injected callable
            last_error = str(exc)
    return {
        "status": "DEAD",
        "status_code": status_code,
        "sample_fields": [],
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="600519")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    result = probe_baidu_daily(code=args.code, timeout=args.timeout, retries=args.retries)
    print(
        f"{result['status']} status_code={result.get('status_code')} "
        f"sample_fields={result.get('sample_fields')} error={result.get('error')}"
    )
    raise SystemExit(0 if result["status"] == "ALIVE" else 1)


if __name__ == "__main__":
    main()
