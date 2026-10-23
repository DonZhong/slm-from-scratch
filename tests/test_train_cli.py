
import subprocess
import sys
from pathlib import Path

import torch

from slm.config import TinyGPTConfig
from slm.model import TinyGPT

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
