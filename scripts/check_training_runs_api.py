
#!/usr/bin/env python3
"""验证训练实验列表 API 能读取 PostgreSQL 中的记录。"""

import json
import os
import sys
from urllib.request import urlopen

BASE_URL = os.environ.get(
    "SLM_BASE_URL",
    "http://127.0.0.1:8080",
).rstrip("/")

EXPECTED_RUN_NAME = os.environ.get(
    "SLM_EXPECTED_RUN_NAME",
    "ci-training-runs-smoke",
)

EXPECTED_DATASET_NAME = os.environ.get(
    "SLM_EXPECTED_DATASET_NAME",
    "synthetic-ci",
)


def get_json(path: str) -> dict:
    """发送 GET 请求，并将 JSON 响应解析为字典。"""
    url = f"{BASE_URL}{path}"

    with urlopen(url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(
                f"GET {path}: HTTP {response.status}"
            )

        return json.load(response)


def main() -> None:
    # 1. 验证后端可以连接数据库。
    health = get_json("/api/db/health")

    if (
        health.get("status") != "ok"
        or health.get("database") != "connected"
    ):
        raise RuntimeError(
            f"Database health check failed: {health}"
        )

    # 2. 获取训练实验列表。
    payload = get_json("/api/training-runs")
    items = payload.get("items")

    if payload.get("status") != "ok":
        raise RuntimeError(
            f"Training runs API failed: {payload}"
        )

    if not isinstance(items, list):
        raise RuntimeError(
            "API field 'items' must be a list"
        )

    if payload.get("count") != len(items):
        raise RuntimeError(
            "API count does not match the returned items"
        )

    # 3. 查找预先插入的测试记录。
    matches = [
        item
        for item in items
        if item.get("name") == EXPECTED_RUN_NAME
    ]

    if not matches:
        raise RuntimeError(
            f"Expected training run not found: "
            f"{EXPECTED_RUN_NAME}"
        )

    run = matches[0]

    # 4. 验证数据库记录的关键字段。
    required_fields = (
        "id",
        "name",
        "model_name",
        "dataset_name",
        "status",
        "created_at",
        "started_at",
        "finished_at",
    )

    missing = [
        field
        for field in required_fields
        if field not in run
    ]

    if missing:
        raise RuntimeError(
            f"Missing fields in training run: {missing}"
        )

    if run["dataset_name"] != EXPECTED_DATASET_NAME:
        raise RuntimeError(
            f"Unexpected dataset: {run['dataset_name']}"
        )

    if run["model_name"] != "TinyGPT":
        raise RuntimeError(
            f"Unexpected model: {run['model_name']}"
        )

    if run["status"] != "created":
        raise RuntimeError(
            f"Unexpected status: {run['status']}"
        )

    print(
        "PASS: training-runs API returned the expected "
        f"database record (id={run['id']})"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
