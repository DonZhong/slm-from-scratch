"""Exercise additive config/diagnostic fields against an isolated development API."""
import json
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("SLM_API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


def request(path, payload=None):
    req = Request(BASE_URL + path,
                  data=json.dumps(payload).encode() if payload is not None else None,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def main():
    config = {"planned_steps": 4, "n_layers": 2, "diagnostics_interval": 2}
    status, created = request("/api/training-runs", {
        "name": f"ci-diagnostics-{time.time_ns()}", "dataset_name": "synthetic-ci", "config": config,
    })
    assert status == 200 or status == 201, (status, created)
    run_id = created["item"]["id"]
    status, runs = request("/api/training-runs")
    assert status == 200
    assert next(run for run in runs["items"] if run["id"] == run_id)["config"] == config
    sample = {"version": 1, "gradient_norm": 0.25, "update_norm": 0.005,
              "layers": [{"name": "embeddings", "activation_mean": -0.1}],
              "attention": {"tokens": ["a"], "weights": [[1.0]], "layer": 0, "head": 0}}
    for step, (validation, speed) in enumerate([(1.2, 100), (1.1, None), (None, 120), (None, None)], 1):
        metric = {"global_step": step, "epoch": 1, "train_loss": 1.0, "learning_rate": 0.001,
                  "val_loss": validation, "tokens_per_second": speed}
        if step != 4:
            metric["diagnostics"] = sample
        status, result = request(f"/api/training-runs/{run_id}/metrics", metric)
        assert status == 201, (status, result)
        assert result["item"]["diagnostics"] == (sample if step != 4 else None)
    status, result = request(f"/api/training-runs/{run_id}/metrics")
    assert status == 200
    assert [item["diagnostics"] for item in result["items"]] == [sample, sample, sample, None]
    metric["global_step"] = 5
    for invalid in [[], "invalid", {"oversized": "x" * 65536}]:
        metric["diagnostics"] = invalid
        status, result = request(f"/api/training-runs/{run_id}/metrics", metric)
        assert status == 400, (status, result)
    status, result = request("/api/training-runs", {"name": "invalid", "dataset_name": "ci", "config": []})
    assert status == 400, (status, result)
    print("PASS: config persistence, all optional-metric SQL variants, diagnostic roundtrip, legacy compatibility and invalid payloads")


if __name__ == "__main__":
    main()
