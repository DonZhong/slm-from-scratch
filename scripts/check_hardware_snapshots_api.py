
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv(
    "SLM_API_BASE_URL",
    "http://127.0.0.1:8080",
)


def request_json(path, method="GET", payload=None):
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        BASE_URL.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as error:
        body = error.read().decode("utf-8")
        return error.code, json.loads(body) if body else {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    # 1. 准备独立的模拟硬件快照。
    sampled_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    host_name = f"ci-hardware-{uuid.uuid4().hex[:12]}"

    payload = {
        "sampled_at": sampled_at,
        "host_name": host_name,
        "training_run_id": None,
        "metrics": {
            "gpu": {
                "name": "CI Test GPU",
                "utilization_percent": 25,
                "memory_used_mb": 1024,
                "memory_total_mb": 4096,
                "temperature_c": 50,
            },
            "cpu": {
                "utilization_percent": 35,
            },
            "memory": {
                "used_mb": 4096,
                "total_mb": 16384,
            },
        },
    }

    # 2. 验证创建快照。
    status, result = request_json(
        "/api/hardware-snapshots",
        method="POST",
        payload=payload,
    )

    require(status == 201, f"POST expected 201, got {status}")
    require(result.get("status") == "ok", "POST status is not ok")

    item = result.get("item")
    require(isinstance(item, dict), "Missing snapshot item")
    require(isinstance(item.get("id"), int), "Missing snapshot ID")
    require(item.get("host_name") == host_name, "Incorrect host_name")
    require(item.get("metrics") == payload["metrics"], "Metrics mismatch")

    snapshot_id = item["id"]

    # 3. 验证最新快照接口。
    status, result = request_json(
        "/api/hardware-snapshots/latest"
    )

    require(status == 200, f"Latest expected 200, got {status}")
    require(result.get("status") == "ok", "Latest query failed")
    require(result.get("count") == 1, "Expected latest count 1")
    require(
        result.get("item", {}).get("id") == snapshot_id,
        "Latest snapshot ID mismatch",
    )

    # 4. 验证历史快照接口。
    status, result = request_json(
        "/api/hardware-snapshots?limit=10"
    )

    require(status == 200, f"History expected 200, got {status}")
    require(result.get("status") == "ok", "History query failed")

    items = result.get("items")
    require(isinstance(items, list), "items must be a list")
    require(
        any(entry.get("id") == snapshot_id for entry in items),
        "Created snapshot missing from history",
    )

    # 5. 验证非法 limit 被拒绝。
    status, result = request_json(
        "/api/hardware-snapshots?limit=abc"
    )

    require(status == 400, f"Invalid limit expected 400, got {status}")
    require(result.get("status") == "error", "Expected error response")

    print("PASS: hardware snapshots API integration checks")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
