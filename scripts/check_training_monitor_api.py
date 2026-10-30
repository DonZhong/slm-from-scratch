"""Monitoring API acceptance; creates records only in an isolated development database."""
import time
from check_training_diagnostics_api import request


def main():
    status, created = request("/api/training-runs", {"name": f"ci-monitor-{time.time_ns()}", "dataset_name": "synthetic-ci", "config": {"planned_steps": 1002}})
    assert status in (200, 201), (status, created)
    run_id = created["item"]["id"]
    base = f"/api/training-runs/{run_id}"
    assert request(base + "/status", {"status": "running"})[0] == 200
    runtime = {"phase": "training", "global_step": 1, "epoch": 1}
    assert request(base + "/runtime", runtime)[0] == 200
    sample = {"version": 1, "gradient_norm": 0.1, "pipeline": {"embeddings": 0.2}, "input_shape": [1, 4]}
    for step in range(1, 1003):
        metric = {"global_step": step, "epoch": 1, "train_loss": 2.0 if step == 1 else 1.0,
                  "learning_rate": 0.001, "tokens_per_second": 0 if step == 1 else None}
        if step == 1:
            metric["diagnostics"] = sample
        if step == 1002:
            metric["val_loss"] = 0.8
        assert request(base + "/metrics", metric)[0] == 201
    status, monitor = request(base + "/monitor")
    assert status == 200, (status, monitor)
    assert monitor["runtime"] == runtime
    summary = monitor["summary"]
    assert summary["total_metrics"] == 1002
    assert summary["latest_step"] == 1002
    assert summary["diagnostics_count"] == 1
    assert summary["average_speed"] == 0
    assert summary["loss_change_percent"] == 50
    assert summary["last_val_loss"] == 0.8
    assert monitor["latest_diagnostic"]["global_step"] == 1
    assert request(base + "/metrics")[1]["items"][0]["global_step"] == 3
    status, diagnostics = request(base + "/diagnostics?at_step=1")
    assert status == 200 and diagnostics["items"][0]["diagnostics"] == sample
    status, page = request(base + "/metrics/history?limit=2")
    assert status == 200 and page["has_more"]
    assert [m["global_step"] for m in page["items"]] == [1001, 1002]
    cursor = page["next_before_step"]
    page = request(base + f"/metrics/history?limit=2&before_step={cursor}")[1]
    assert [m["global_step"] for m in page["items"]] == [999, 1000]
    for path in ["/metrics/history?limit=1001", "/metrics/history?before_step=-1", "/diagnostics?at_step=nope"]:
        assert request(base + path)[0] == 400
    assert request(base + "/runtime", {**runtime, "phase": "invalid"})[0] == 400
    assert request("/api/training-runs/999999999/monitor")[0] == 404
    assert request("/api/training-runs/999999999/diagnostics")[0] == 404
    assert request(base + "/runtime", {**runtime, "phase": "completed"})[0] == 200
    assert request(base + "/status", {"status": "completed"})[0] == 200
    assert request(base + "/runtime", runtime)[0] == 409
    status, empty = request("/api/training-runs", {"name": f"ci-monitor-empty-{time.time_ns()}", "dataset_name": "synthetic-ci"})
    empty_base = f"/api/training-runs/{empty['item']['id']}"
    assert request(empty_base + "/monitor")[1]["summary"]["total_metrics"] == 0
    assert request(empty_base + "/metrics/history")[1]["items"] == []
    print("PASS: runtime, whole-run statistics, diagnostics outside latest 1000 metrics, keyset history, empty runs and validation errors")


if __name__ == "__main__":
    main()
