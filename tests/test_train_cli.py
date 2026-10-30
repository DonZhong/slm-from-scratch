
import subprocess
import sys
import os
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from slm.config import TinyGPTConfig
from slm.model import TinyGPT


def test_train_cli_reports_config_and_sampled_diagnostics():
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            result = {"status": "ok", "item": {"id": 42}}
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run([
            sys.executable, str(root / "scripts/train.py"), "--max-steps", "3",
            "--batch-size", "8", "--diagnostics-interval", "3",
            "--api-base-url", f"http://127.0.0.1:{server.server_port}",
        ], cwd=root, env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    config = received[0][1]["config"]
    assert config["planned_steps"] == 3
    assert config["parameter_count"] > 0
    assert config["context_length"] == 16
    metrics = [payload for path, payload in received if path.endswith("/metrics")]
    assert [metric["global_step"] for metric in metrics] == [1, 2, 3]
    assert "diagnostics" in metrics[0]
    assert "diagnostics" not in metrics[1]
    assert "diagnostics" in metrics[2]
    assert metrics[2]["val_loss"] is not None
    assert received[-1][1] == {"status": "completed"}

def test_train_cli_respects_max_steps() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train.py"

    result = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--epochs", "3",
            "--batch-size", "8",
            "--learning-rate", "0.0005",
            "--max-steps", "2",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "total steps: 2" in result.stdout
    assert "epoch=02" not in result.stdout



def test_train_cli_checkpoint_roundtrip(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train.py"

    checkpoint_path = tmp_path / "tinygpt-test.pt"

    result = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--epochs", "3",
            "--batch-size", "8",
            "--learning-rate", "0.0005",
            "--max-steps", "2",
            "--checkpoint-path", str(checkpoint_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert checkpoint_path.is_file()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["global_step"] == 2
    assert checkpoint["epoch"] == 1
    assert checkpoint["token_to_id"]["<unk>"] == 0

    assert "optimizer_state_dict" in checkpoint
    assert "val_loss" in checkpoint

    config = TinyGPTConfig(**checkpoint["config"])

    restored_model = TinyGPT(config)

    restored_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    restored_model.eval()

    input_ids = torch.tensor(
        [[0, 1]],
        dtype=torch.long,
    )

    with torch.inference_mode():
        logits = restored_model(input_ids)

    assert logits.shape == (1, 2, config.vocab_size)
    assert torch.isfinite(logits).all()
