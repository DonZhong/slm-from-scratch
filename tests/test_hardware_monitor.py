
import io
import json
from urllib.error import HTTPError

import pytest

import scripts.hardware_monitor as monitor


def test_parse_nvidia_smi_output():
    output = (
        "NVIDIA GeForce GTX 1650 Ti, "
        "25, 1024, 4096, 48, 12.5, 50, 1200, 3000\n"
    )

    devices = monitor.parse_nvidia_smi_output(output)

    assert len(devices) == 1
    assert devices[0]["name"] == "NVIDIA GeForce GTX 1650 Ti"
    assert devices[0]["utilization_percent"] == 25
    assert devices[0]["memory_used_mb"] == 1024
    assert devices[0]["memory_total_mb"] == 4096
    assert devices[0]["temperature_c"] == 48
    assert devices[0]["power_draw_w"] == 12.5
    assert devices[0]["power_limit_w"] == 50
    assert devices[0]["graphics_clock_mhz"] == 1200
    assert devices[0]["memory_clock_mhz"] == 3000


def test_parse_nvidia_smi_unavailable_values():
    output = (
        "NVIDIA GPU, N/A, N/A, 4096, N/A, "
        "N/A, N/A, N/A, N/A\n"
    )

    devices = monitor.parse_nvidia_smi_output(output)

    assert len(devices) == 1
    assert devices[0]["utilization_percent"] is None
    assert devices[0]["memory_used_mb"] is None
    assert devices[0]["temperature_c"] is None


def test_collect_gpu_metrics_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        monitor.shutil,
        "which",
        lambda _: None,
    )

    result = monitor.collect_gpu_metrics()

    assert result["available"] is False
    assert result["device_count"] == 0
    assert result["devices"] == []
    assert "error" in result


def test_send_snapshot_success(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"status":"ok","item":{"id":7}}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(
            request.data.decode("utf-8")
        )
        return FakeResponse()

    monkeypatch.setattr(monitor, "urlopen", fake_urlopen)

    payload = {
        "sampled_at": "2026-10-25T12:00:00Z",
        "host_name": "test-host",
        "training_run_id": None,
        "metrics": {"cpu": {"utilization_percent": 10}},
    }

    result = monitor.send_snapshot(
        "http://127.0.0.1:8080/",
        payload,
        timeout=2,
        retries=0,
    )

    assert captured["url"] == (
        "http://127.0.0.1:8080/api/hardware-snapshots"
    )
    assert captured["timeout"] == 2
    assert captured["payload"] == payload
    assert result["item"]["id"] == 7


def test_send_snapshot_does_not_retry_client_error(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"message":"invalid payload"}'),
        )

    monkeypatch.setattr(monitor, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Backend rejected"):
        monitor.send_snapshot(
            "http://127.0.0.1:8080",
            {"metrics": {}},
            retries=3,
        )

    # Client errors are not transient and must not be retried.
    assert len(calls) == 1


def test_hardware_collector_returns_expected_sections():
    collector = monitor.HardwareCollector(disk_paths=["/"])

    metrics = collector.collect_metrics()

    assert {
        "cpu",
        "memory",
        "disk",
        "network",
        "gpu",
        "system",
    }.issubset(metrics.keys())

    assert "per_core_utilization_percent" in metrics["cpu"]
    assert "swap" in metrics["memory"]
    assert "mounts" in metrics["disk"]
    assert "io" in metrics["disk"]
    assert "interfaces" in metrics["network"]
    assert isinstance(metrics["gpu"]["available"], bool)
    assert metrics["system"]["collector_version"] == "1.0.0"
