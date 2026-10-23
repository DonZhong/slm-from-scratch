from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [B, T, V], got {tuple(logits.shape)}"
        )

    if targets.ndim != 2:
        raise ValueError(
            f"targets must have shape [B, T], got {tuple(targets.shape)}"
        )

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits and targets must agree on batch and sequence dimensions: "
            f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
        )

    batch_size, sequence_length, vocab_size = logits.shape

    return F.cross_entropy(
        logits.reshape(batch_size * sequence_length, vocab_size),
        targets.reshape(batch_size * sequence_length),
    )


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    model.train()

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    loss = language_model_loss(logits, targets)

    loss.backward()
    optimizer.step()

    return loss.detach()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    was_training = model.training

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    try:
        with torch.inference_mode():
            for input_ids, targets in loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)

                logits = model(input_ids)

                loss = language_model_loss(
                    logits,
                    targets,
                )

                token_count = targets.numel()

                total_loss += loss.item() * token_count
                total_tokens += token_count
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("Evaluation loader is empty")

    return total_loss / total_tokens
