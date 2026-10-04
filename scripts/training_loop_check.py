from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from slm.config import TinyGPTConfig
from slm.data import NextTokenDataset
from slm.model import TinyGPT
from slm.tokenizer import CharacterTokenizer


def compute_loss(
    model: TinyGPT,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids)

    batch_size, sequence_length, vocab_size = (
        logits.shape
    )

    return F.cross_entropy(
        logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        ),
        target_ids.reshape(
            batch_size * sequence_length
        ),
    )


def main():
    torch.manual_seed(42)

    text = (
        "hello tiny gpt\n"
        * 32
    )

    tokenizer = CharacterTokenizer.from_text(text)

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        dropout=0.0,
    )

    dataset = NextTokenDataset(
        token_ids=tokenizer.encode(text),
        context_length=config.context_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TinyGPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )

    epochs = 20
    global_step = 0

    first_epoch_loss = None
    final_epoch_loss = None

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_tokens = 0

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = compute_loss(
                model,
                input_ids,
                target_ids,
            )

            loss.backward()

            optimizer.step()

            token_count = target_ids.numel()

            total_loss += (
                loss.item() * token_count
            )

            total_tokens += token_count
            global_step += 1

        average_loss = (
            total_loss / total_tokens
        )

        if first_epoch_loss is None:
            first_epoch_loss = average_loss

        final_epoch_loss = average_loss

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch:02d} "
                f"step={global_step:04d} "
                f"loss={average_loss:.6f}"
            )

    print()
    print("device:", device)
    print("vocab size:", tokenizer.vocab_size)
    print("dataset size:", len(dataset))
    print("batches per epoch:", len(loader))
    print("first epoch loss:", first_epoch_loss)
    print("final epoch loss:", final_epoch_loss)

    print(
        "loss decreased:",
        final_epoch_loss < first_epoch_loss,
    )

    checkpoint_dir = Path(
        "/mnt/d/SLM/checkpoints/slm-from-scratch"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir / "tinygpt-toy-v1.pt"
    )

    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "token_to_id": tokenizer.token_to_id,
        "epoch": epochs,
        "global_step": global_step,
        "loss": final_epoch_loss,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print()
    print("checkpoint path:", checkpoint_path)

    print(
        "checkpoint exists:",
        checkpoint_path.exists(),
    )

    print(
        "checkpoint size bytes:",
        checkpoint_path.stat().st_size,
    )


if __name__ == "__main__":
    main()