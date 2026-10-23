import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from slm.config import TinyGPTConfig
from slm.data import NextTokenDataset, split_text
from slm.model import TinyGPT
from slm.tokenizer import CharacterTokenizer
from slm.training import evaluate, train_step
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

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")

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

    if len(text) <= 16:
        raise ValueError(
            "Training text must contain more than 16 characters."
        )

    train_text, val_text = split_text(
        text,
        train_fraction=0.9,
        context_length=16,
    )

    tokenizer = CharacterTokenizer.from_text(train_text)

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        dropout=0.0,
    )

    train_dataset = NextTokenDataset(
        token_ids=tokenizer.encode(train_text),
        context_length=config.context_length,
    )

    val_dataset = NextTokenDataset(
        token_ids=tokenizer.encode(val_text),
        context_length=config.context_length,
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
    print("vocab size:", tokenizer.vocab_size)
    print("train characters:", len(train_text))
    print("val characters:", len(val_text))
    print("train dataset size:", len(train_dataset))
    print("val dataset size:", len(val_dataset))
    print("batches per epoch:", len(loader))

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_tokens = 0

        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            loss = train_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
            )

            token_count = targets.numel()

            total_loss += loss.item() * token_count
            total_tokens += token_count
            global_step += 1

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        average_loss = total_loss / total_tokens

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
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



if __name__ == "__main__":
    main()
