import json
import math
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv(
    "SLM_API_BASE_URL",
    "http://127.0.0.1:8080",
)


def get_json(path):
    request = Request(
        BASE_URL.rstrip("/") + path,
        headers={"Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        body = error.read().decode("utf-8")
        return error.code, json.loads(body) if body else {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_number(actual, expected, label):
    require(
        isinstance(actual, (int, float))
        and math.isclose(
            actual, expected, rel_tol=1e-8, abs_tol=1e-10
        ),
        f"{label}: expected {expected}, got {actual}",
    )


def main():
    # 1. 从现有 API 查找 CI 创建的测试实验
    status, runs = get_json("/api/training-runs")
    require(status == 200, f"Training runs HTTP {status}")
    require(runs.get("status") == "ok", "Training runs API failed")

    run_by_name = {
        item.get("name"): item
        for item in runs.get("items", [])
    }

    populated = run_by_name.get("ci-training-metrics-query-smoke")
    empty = run_by_name.get("ci-training-metrics-query-empty")

    require(populated is not None, "Populated test run not found")
    require(empty is not None, "Empty test run not found")

    run_id = populated["id"]

    # 2. 验证正常查询、指标字段和 global_step 排序
    status, result = get_json(
        f"/api/training-runs/{run_id}/metrics"
    )

    require(status == 200, f"Metrics API HTTP {status}")
    require(result.get("status") == "ok", "Metrics API failed")
    require(result.get("run_id") == run_id, "Incorrect run_id")

    items = result.get("items")
    require(isinstance(items, list), "items must be a list")
    require(result.get("count") == 3, "Expected 3 metric records")
    require(len(items) == 3, "Incorrect number of metric items")

    require(
        [item.get("global_step") for item in items] == [1, 2, 3],
        "Metrics must be ordered by global_step ascending",
    )

    fields = (
        "train_loss",
        "val_loss",
        "learning_rate",
        "tokens_per_second",
    )
    for item in items:
        for field in fields:
            require(field in item, f"Missing field: {field}")

    check_number(items[0]["train_loss"], 3.1, "step 1 train_loss")
    check_number(items[0]["val_loss"], 3.0, "step 1 val_loss")
    check_number(items[0]["learning_rate"], 0.0003, "step 1 learning_rate")
    check_number(items[0]["tokens_per_second"], 120.5, "step 1 tokens_per_second")

    check_number(items[1]["train_loss"], 2.9, "step 2 train_loss")
    check_number(items[1]["val_loss"], 2.8, "step 2 val_loss")
    require(items[1]["tokens_per_second"] is None, "Expected null tokens_per_second")

    check_number(items[2]["train_loss"], 2.7, "step 3 train_loss")
    require(items[2]["val_loss"] is None, "Expected null val_loss")
    check_number(items[2]["tokens_per_second"], 115.0, "step 3 tokens_per_second")

    # 3. 验证没有指标的实验返回空数组，而不是报错
    status, result = get_json(
        f"/api/training-runs/{empty['id']}/metrics"
    )
    require(status == 200, f"Empty metrics HTTP {status}")
    require(result.get("count") == 0, "Expected zero metrics")
    require(result.get("items") == [], "Expected an empty items array")

    # 4. 验证错误 ID 的 HTTP 状态码
    status, result = get_json(
        "/api/training-runs/not-a-number/metrics"
    )
    require(status == 400, f"Invalid ID should return 400, got {status}")

    status, result = get_json(
        "/api/training-runs/999999999/metrics"
    )
    require(status == 404, f"Missing run should return 404, got {status}")

    print("PASS: training-metrics query API integration checks")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)