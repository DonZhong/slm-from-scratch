
"""Collect host hardware metrics and upload snapshots to the SLM backend."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psutil


COLLECTOR_VERSION = "1.0.0"
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

NVIDIA_FIELDS = (
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.gr",
    "clocks.mem",
)


def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp in ISO 8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def optional_float(value: str) -> float | None:
    """Parse a numeric hardware value; unavailable readings become None."""
    text = value.strip()

    if text.lower() in {
        "",
        "n/a",
        "[not supported]",
        "not supported",
    }:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def optional_int(value: str) -> int | None:
    number = optional_float(value)
    return int(number) if number is not None else None


def parse_nvidia_smi_output(output: str) -> list[dict]:
    """Parse nvidia-smi CSV output without units."""
    devices = []

    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        row = [field.strip() for field in row]

        if len(row) != len(NVIDIA_FIELDS):
            continue

        name = row[0]

        if not name:
            continue

        devices.append(
            {
                "name": name,
                "utilization_percent": optional_float(row[1]),
                "memory_used_mb": optional_int(row[2]),
                "memory_total_mb": optional_int(row[3]),
                "temperature_c": optional_float(row[4]),
                "power_draw_w": optional_float(row[5]),
                "power_limit_w": optional_float(row[6]),
                "graphics_clock_mhz": optional_int(row[7]),
                "memory_clock_mhz": optional_int(row[8]),
            }
        )

    return devices


def collect_gpu_metrics(timeout: float = 3.0) -> dict:
    """Read GPU telemetry through NVIDIA's nvidia-smi utility."""
    executable = shutil.which("nvidia-smi")

    if executable is None:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "error": "nvidia-smi is not available",
        }

    query = ",".join(NVIDIA_FIELDS)

    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "error": str(exc),
        }

    if result.returncode != 0:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "error": result.stderr.strip()
            or f"nvidia-smi exited with code {result.returncode}",
        }

    devices = parse_nvidia_smi_output(result.stdout)

    metrics = {
        "available": bool(devices),
        "device_count": len(devices),
        "devices": devices,
    }

    if devices:
        # Keep the first GPU's metrics at the top level for convenient
        # dashboard access while retaining details for every GPU.
        metrics.update(devices[0])
    else:
        metrics["error"] = "No GPU telemetry was returned"

    return metrics


def safe_load_average() -> dict | None:
    """Read Linux load averages where supported."""
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return None

    return {
        "1m": round(one, 2),
        "5m": round(five, 2),
        "15m": round(fifteen, 2),
    }


def collect_cpu_temperatures() -> list[dict]:
    """Read temperature sensors when the OS exposes them."""
    getter = getattr(psutil, "sensors_temperatures", None)

    if getter is None:
        return []

    try:
        sensor_groups = getter(fahrenheit=False)
    except (OSError, NotImplementedError):
        return []

    readings = []

    for sensor_name, entries in sensor_groups.items():
        for entry in entries:
            if entry.current is None:
                continue

            readings.append(
                {
                    "sensor": sensor_name,
                    "label": entry.label or "",
                    "temperature_c": round(float(entry.current), 2),
                }
            )

    return readings


def counter_rate(
    current: int | None,
    previous: int | None,
    elapsed_seconds: float | None,
) -> float | None:
    """Calculate a non-negative counter rate in units per second."""
    if current is None or previous is None:
        return None

    if elapsed_seconds is None or elapsed_seconds <= 0:
        return None

    delta = current - previous

    if delta < 0:
        # Counters may reset after an interface or device restart.
        return None

    return round(delta / elapsed_seconds, 2)


def bytes_to_mb(value: int) -> float:
    return round(value / MIB, 2)


def bytes_to_gb(value: int) -> float:
    return round(value / GIB, 2)


class HardwareCollector:
    """Collect hardware metrics and calculate rates between samples."""

    def __init__(
        self,
        host_name: str | None = None,
        disk_paths: list[str] | None = None,
    ):
        self.host_name = host_name or socket.gethostname()

        if disk_paths is None:
            disk_paths = ["/"]

            if os.path.exists("/mnt/d"):
                disk_paths.append("/mnt/d")

        self.disk_paths = disk_paths

        self._last_sample_monotonic = None
        self._previous_disk_io = None
        self._previous_network_io = None

    def collect_disk_usage(self) -> list[dict]:
        """Collect capacity and usage for configured mount paths."""
        mounts = []

        for path in self.disk_paths:
            try:
                usage = psutil.disk_usage(path)
            except OSError:
                continue

            mounts.append(
                {
                    "path": path,
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                    "total_gb": bytes_to_gb(usage.total),
                    "used_gb": bytes_to_gb(usage.used),
                    "free_gb": bytes_to_gb(usage.free),
                    "percent": round(float(usage.percent), 2),
                }
            )

        return mounts

    def collect_metrics(self) -> dict:
        """Collect one hardware sample and calculate interval rates."""
        # A short blocking interval produces a meaningful CPU reading
        # instead of relying on the uninitialized first non-blocking sample.
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        cpu_percent = (
            round(sum(per_core) / len(per_core), 2)
            if per_core
            else 0.0
        )

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()

        now = time.monotonic()

        elapsed = (
            now - self._last_sample_monotonic
            if self._last_sample_monotonic is not None
            else None
        )

        disk_read = (
            int(disk_io.read_bytes) if disk_io is not None else None
        )
        disk_write = (
            int(disk_io.write_bytes) if disk_io is not None else None
        )

        previous_disk = self._previous_disk_io or {}

        disk_io_metrics = {
            "read_bytes_total": disk_read,
            "write_bytes_total": disk_write,
            "read_bytes_per_second": counter_rate(
                disk_read,
                previous_disk.get("read"),
                elapsed,
            ),
            "write_bytes_per_second": counter_rate(
                disk_write,
                previous_disk.get("write"),
                elapsed,
            ),
        }

        sent = (
            int(network_io.bytes_sent)
            if network_io is not None
            else None
        )
        received = (
            int(network_io.bytes_recv)
            if network_io is not None
            else None
        )

        previous_network = self._previous_network_io or {}

        interfaces = []

        for name, counters in (
            psutil.net_io_counters(pernic=True) or {}
        ).items():
            interfaces.append(
                {
                    "name": name,
                    "bytes_sent_total": int(counters.bytes_sent),
                    "bytes_recv_total": int(counters.bytes_recv),
                    "packets_sent_total": int(counters.packets_sent),
                    "packets_recv_total": int(counters.packets_recv),
                    "errors_in": int(counters.errin),
                    "errors_out": int(counters.errout),
                    "drops_in": int(counters.dropin),
                    "drops_out": int(counters.dropout),
                }
            )

        network_metrics = {
            "bytes_sent_total": sent,
            "bytes_recv_total": received,
            "bytes_sent_per_second": counter_rate(
                sent,
                previous_network.get("sent"),
                elapsed,
            ),
            "bytes_recv_per_second": counter_rate(
                received,
                previous_network.get("received"),
                elapsed,
            ),
            "interfaces": interfaces,
        }

        # Store counters for the next sample.
        self._previous_disk_io = {
            "read": disk_read,
            "write": disk_write,
        }
        self._previous_network_io = {
            "sent": sent,
            "received": received,
        }
        self._last_sample_monotonic = now

        return {
            "cpu": {
                "utilization_percent": cpu_percent,
                "per_core_utilization_percent": [
                    round(float(value), 2) for value in per_core
                ],
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "load_average": safe_load_average(),
                "temperature_sensors": collect_cpu_temperatures(),
            },
            "memory": {
                "used_mb": bytes_to_mb(memory.used),
                "total_mb": bytes_to_mb(memory.total),
                "available_mb": bytes_to_mb(memory.available),
                "percent": round(float(memory.percent), 2),
                "swap": {
                    "used_mb": bytes_to_mb(swap.used),
                    "total_mb": bytes_to_mb(swap.total),
                    "percent": round(float(swap.percent), 2),
                },
            },
            "disk": {
                "mounts": self.collect_disk_usage(),
                "io": disk_io_metrics,
            },
            "network": network_metrics,
            "gpu": collect_gpu_metrics(),
            "system": {
                "hostname": self.host_name,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "collector_version": COLLECTOR_VERSION,
            },
        }

    def build_payload(
        self,
        training_run_id: int | None = None,
    ) -> dict:
        """Build the request body accepted by the existing backend API."""
        return {
            "sampled_at": utc_timestamp(),
            "host_name": self.host_name,
            "training_run_id": training_run_id,
            "metrics": self.collect_metrics(),
        }


def send_snapshot(
    base_url: str,
    payload: dict,
    timeout: float = 5.0,
    retries: int = 3,
) -> dict:
    """Upload a snapshot, retrying transient network and server failures."""
    url = base_url.rstrip("/") + "/api/hardware-snapshots"

    body = json.dumps(
        payload,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    last_error = None

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                response_body = response.read().decode("utf-8")

            if status != 201:
                raise RuntimeError(
                    f"Expected HTTP 201, received HTTP {status}"
                )

            return json.loads(response_body) if response_body else {}

        except HTTPError as exc:
            response_body = exc.read().decode(
                "utf-8", errors="replace"
            )

            message = (
                f"HTTP {exc.code}: {response_body or exc.reason}"
            )

            # Invalid payloads and other client errors will not improve
            # by retrying. HTTP 429 is treated as a transient response.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(
                    f"Backend rejected the snapshot: {message}"
                ) from exc

            last_error = RuntimeError(message)

        except (URLError, TimeoutError, OSError) as exc:
            last_error = RuntimeError(str(exc))

        if attempt < retries:
            delay = min(2**attempt, 8)
            logging.warning(
                "Upload attempt %d/%d failed; retrying in %ds: %s",
                attempt + 1,
                retries + 1,
                delay,
                last_error,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Snapshot upload failed after {retries + 1} attempts: "
        f"{last_error}"
    )


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer"
        ) from exc

    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")

    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a number"
        ) from exc

    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        )

    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect and upload SLM hardware metrics."
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "SLM_API_BASE_URL",
            "http://127.0.0.1:8080",
        ),
        help="Backend base URL",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=5.0,
        help="Seconds between samples (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=5.0,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for transient upload failures",
    )
    parser.add_argument(
        "--training-run-id",
        type=positive_int,
        default=None,
        help="Optional existing training run ID",
    )
    parser.add_argument(
        "--host-name",
        default=None,
        help="Override the detected host name",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect and upload one snapshot, then exit",
    )

    args = parser.parse_args(argv)

    if args.retries < 0:
        parser.error("--retries must be zero or greater")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    collector = HardwareCollector(host_name=args.host_name)

    try:
        while True:
            cycle_started = time.monotonic()

            try:
                payload = collector.build_payload(
                    training_run_id=args.training_run_id
                )

                response = send_snapshot(
                    base_url=args.base_url,
                    payload=payload,
                    timeout=args.timeout,
                    retries=args.retries,
                )

                metrics = payload["metrics"]
                cpu = metrics["cpu"]["utilization_percent"]
                memory = metrics["memory"]["percent"]
                gpu = metrics["gpu"].get("utilization_percent")

                gpu_text = (
                    f"{gpu:.1f}%"
                    if gpu is not None
                    else "N/A"
                )

                item = response.get("item", {})

                logging.info(
                    "Snapshot uploaded: id=%s CPU=%.1f%% "
                    "GPU=%s RAM=%.1f%%",
                    item.get("id", "unknown"),
                    cpu,
                    gpu_text,
                    memory,
                )

            except Exception as exc:
                logging.error("Collection/upload failed: %s", exc)

                if args.once:
                    return 1

            if args.once:
                return 0

            elapsed = time.monotonic() - cycle_started
            time.sleep(max(0.0, args.interval - elapsed))

    except KeyboardInterrupt:
        logging.info("Hardware collector stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
