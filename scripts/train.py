import argparse
import math
from dataclasses import asdict
from pathlib import Path
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader

from slm.config import TinyGPTConfig
from slm.data import NextTokenDataset, split_text
from slm.model import TinyGPT
from slm.tokenizer import CharacterTokenizer
from slm.training import evaluate, train_step
from slm.diagnostics import diagnostic_train_step


def post_api_json(
    base_url: str,
    path: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """向后端发送 JSON POST 请求，并检查响应。"""

    body = json.dumps(payload).encode("utf-8")

    request = Request(
        url=f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=5) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as exc:
        detail = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"API POST {path} failed "
            f"(HTTP {exc.code}): {detail}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Cannot reach training API {path}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected API response from {path}"
        )

    if result.get("status") != "ok":
        raise RuntimeError(
            f"API returned an error for {path}: {result}"
        )

    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TinyGPT language model."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of samples per batch.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to a UTF-8 training text file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of optimizer steps.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to save the training checkpoint.",
    )

    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help=(
            "Optional training API base URL, "
            "e.g. http://127.0.0.1:8080"
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this training run.",
    )

    parser.add_argument(
    "--context-length",
    type=int,
    default=16,
    help="Number of context tokens.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window stride for the dataset.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability.",
    parser.add_argument(
        "--diagnostics-interval", type=int, default=0,
        help="Capture model internals every N steps (0 disables; includes short training text excerpts).",
    )
    args = parser.parse_args()

    if args.diagnostics_interval < 0:
        parser.error("--diagnostics-interval must be zero or greater")

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    if args.context_length <= 0:
        parser.error("--context-length must be positive")

    if args.stride <= 0:
        parser.error("--stride must be positive")

    if (
        not math.isfinite(args.dropout)
        or not 0.0 <= args.dropout < 1.0
    ):
        parser.error("--dropout must be finite and in [0, 1)")

    return args



def main() -> None:
    args = parse_args()

    if args.checkpoint_path is not None:
        if args.checkpoint_path.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {args.checkpoint_path}"
            )

    torch.manual_seed(42)

    # Temporary toy corpus for training smoke tests.

    if args.data_path is None:
        text = "hello tiny gpt\n" * 32
        print("dataset source: built-in toy text")
    else:
        if not args.data_path.is_file():
            raise FileNotFoundError(
                f"Training file not found: {args.data_path}"
            )

        text = args.data_path.read_text(encoding="utf-8")
        print("dataset source:", args.data_path)

    if len(text) <= args.context_length:
        raise ValueError(
            "Training text must contain more characters "
            "than context_length."
        )

    train_text, val_text = split_text(
        text,
        train_fraction=0.9,
        context_length=args.context_length,
    )

    tokenizer = CharacterTokenizer.from_text(train_text)

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        dropout=args.dropout,
    )

    train_dataset = NextTokenDataset(
        token_ids=tokenizer.encode(train_text),
        context_length=config.context_length,
        stride=args.stride,
    )

    val_dataset = NextTokenDataset(
        token_ids=tokenizer.encode(val_text),
        context_length=config.context_length,
        stride=args.stride,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TinyGPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    global_step = 0
    first_epoch_loss = None
    final_epoch_loss = None

    print("device:", device)
    print("epochs:", args.epochs)
    print("batch size:", args.batch_size)
    print("learning rate:", args.learning_rate)

    print("context length:", config.context_length)
    print("dataset stride:", args.stride)
    print("dropout:", config.dropout)


    print("vocab size:", tokenizer.vocab_size)
    print("train characters:", len(train_text))
    print("val characters:", len(val_text))
    print("train dataset size:", len(train_dataset))
    print("val dataset size:", len(val_dataset))
    print("batches per epoch:", len(loader))

    # 训练实验 ID；未启用 API 时保持 None
    run_id: int | None = None

    if args.api_base_url:
        # 1. 确定数据集名称
        if args.data_path is None:
            dataset_name = "built-in-toy-text"
        else:
            dataset_name = str(args.data_path)

        # 2. 生成本次实验名称
        run_name = args.run_name or (
            "tinygpt-"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        # 3. 请求后端创建实验
        created = post_api_json(
            args.api_base_url,
            "/api/training-runs",
            {
                "name": run_name,
                "model_name": "TinyGPT",
                "dataset_name": dataset_name,
                "config": {
                    **asdict(config),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "planned_steps": min(args.epochs * len(loader), args.max_steps or args.epochs * len(loader)),
                    "batches_per_epoch": len(loader),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "device": str(device),
                    "diagnostics_interval": args.diagnostics_interval,
                },
            },
        )

        # 4. 读取数据库生成的实验 ID
        item = created.get("item")

        if not isinstance(item, dict):
            raise RuntimeError(
                "Training API did not return a valid item"
            )

        created_id = item.get("id")

        if (
            not isinstance(created_id, int)
            or isinstance(created_id, bool)
            or created_id <= 0
        ):
            raise RuntimeError(
                "Training API returned an invalid run ID"
            )

        run_id = created_id

        # 5. 将实验状态改为 running
        post_api_json(
            args.api_base_url,
            f"/api/training-runs/{run_id}/status",
            {"status": "running"},
        )

        print("training run id:", run_id)
        print("training run name:", run_name)
        print("training run status: running")


    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_tokens = 0
        deferred_metric = None

        for batch_index, (input_ids, targets) in enumerate(
            loader, start=1
        ):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            token_count = targets.numel()

            # 1. 测量本次训练 step 的耗时
            step_started = time.perf_counter()

            diagnostics = None
            capture_diagnostics = args.diagnostics_interval > 0 and (
                global_step == 0
                or (global_step + 1) % args.diagnostics_interval == 0
                or batch_index == len(loader)
                or global_step + 1 == args.max_steps
            )
            if capture_diagnostics:
                loss, diagnostics = diagnostic_train_step(model, optimizer, input_ids, targets, tokenizer)
            else:
                loss = train_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    targets=targets,
                )

            # item() 将 loss 转换为 Python 数值，
            # 同时确保需要读取的 GPU 计算结果已经可用
            step_loss = float(loss.item())

            step_duration = max(
                time.perf_counter() - step_started,
                1e-9,
            )

            tokens_per_second = (
                token_count / step_duration
            )

            # 2. 累积当前 epoch 的平均训练损失
            total_loss += step_loss * token_count
            total_tokens += token_count
            global_step += 1

            # 3. 读取当前优化器学习率
            current_lr = float(
                optimizer.param_groups[0]["lr"]
            )

            # 4. 整理当前 step 的真实训练指标
            metric = {
                "global_step": global_step,
                "epoch": epoch,
                "train_loss": step_loss,
                "learning_rate": current_lr,
                "tokens_per_second": tokens_per_second,
                "val_loss": None,
            }
            if diagnostics is not None:
                metric["diagnostics"] = diagnostics


            is_last_batch = batch_index == len(loader)

            hit_step_limit = (
                args.max_steps is not None
                and global_step >= args.max_steps
            )

            # 5. 将指标写入后端
            if args.api_base_url and run_id is not None:
                if is_last_batch or hit_step_limit:
                    # 暂存最后一步，等验证损失算出后一起写入
                    deferred_metric = metric
                else:
                    post_api_json(
                        args.api_base_url,
                        f"/api/training-runs/{run_id}/metrics",
                        metric,
                    )

            # 6. 达到最大 step 数后结束本轮训练
            if hit_step_limit:
                break



        average_loss = total_loss / total_tokens

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        # 最后一步连同当前 epoch 的验证损失一起写入
        if (
            args.api_base_url
            and run_id is not None
            and deferred_metric is not None
        ):
            deferred_metric["val_loss"] = val_loss

            post_api_json(
                args.api_base_url,
                f"/api/training-runs/{run_id}/metrics",
                deferred_metric,
            )

        if first_epoch_loss is None:
            first_epoch_loss = average_loss

        final_epoch_loss = average_loss

        print(
            f"epoch={epoch:02d} "
            f"step={global_step:04d} "
            f"train_loss={average_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print("training completed")
    print("total steps:", global_step)

    if args.max_steps is not None:
        print("loss trend: not evaluated in step-limited run")
    elif args.epochs > 1:
        print(
            "loss decreased:",
            final_epoch_loss < first_epoch_loss,
        )

    if args.checkpoint_path is None:
        print("checkpoint: not saved")
    else:
        checkpoint_path = args.checkpoint_path
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "token_to_id": tokenizer.token_to_id,
            "epoch": epoch,
            "global_step": global_step,
            "loss": final_epoch_loss,
            "val_loss": val_loss,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print("checkpoint path:", checkpoint_path)
        print("checkpoint exists:", checkpoint_path.exists())
        print("checkpoint bytes:", checkpoint_path.stat().st_size)

    # 训练和 checkpoint 保存均成功后，标记实验完成
    if args.api_base_url and run_id is not None:
        post_api_json(
            args.api_base_url,
            f"/api/training-runs/{run_id}/status",
            {"status": "completed"},
        )

        print("training run status: completed")


if __name__ == "__main__":
    main()
