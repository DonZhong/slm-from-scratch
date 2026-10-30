"""Bounded snapshots from an actual TinyGPT optimizer step.

No extra forward pass is performed: hooks observe the training pass itself.
Snapshots contain a short text excerpt and are intended for local experiments.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import torch

from slm.model import TinyGPT
from slm.tokenizer import CharacterTokenizer
from slm.training import train_step


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def diagnostic_train_step(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    tokenizer: CharacterTokenizer,
) -> tuple[torch.Tensor, dict]:
    """Capture activation/attention before backward, gradients and real updates.

    Norms aggregate every trainable parameter within a top-level stage. Attention
    is head 0 of block 0, before attention dropout, for sample 0 (max 16 tokens).
    Prediction is at the last position of sample 0, before the optimizer update.
    """
    snapshots: dict[str, dict] = {}
    observed: dict = {}
    handles = []
    module_started = {}
    module_ms = {}
    timings = {}

    def start_module(name):
        def start(_module, _inputs):
            if input_ids.is_cuda:
                torch.cuda.synchronize(input_ids.device)
            module_started[name] = time.perf_counter()
        return start
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    def activation_hook(name):
        def capture(_module, _inputs, output):
            if input_ids.is_cuda:
                torch.cuda.synchronize(input_ids.device)
            module_ms[name] = (time.perf_counter() - module_started[name]) * 1000
            observed.setdefault("shapes", {})[name] = list(output.shape)
            values = output.detach().float()
            snapshots[name] = {
                "name": name,
                "activation_mean": finite(values.mean().item()),
                "activation_std": finite(values.std(unbiased=False).item()),
            }
        return capture

    def logits_hook(_module, _inputs, output):
        logits = output.detach().float()
        probabilities = logits[0, -1].softmax(dim=-1)
        top_probabilities, top_ids = probabilities.topk(min(5, probabilities.numel()))
        observed["prediction"] = {
            "context": tokenizer.decode(input_ids[0].tolist()),
            "target": tokenizer.decode([targets[0, -1].item()]),
            "top_tokens": [
                {"token": tokenizer.decode([token_id]), "probability": finite(probability)}
                for token_id, probability in zip(top_ids.tolist(), top_probabilities.tolist())
            ],
        }
        observed["token_accuracy"] = finite((logits.argmax(-1) == targets).float().mean().item())
        observed["prediction_entropy"] = finite(
            -(probabilities * probabilities.clamp_min(1e-30).log()).sum().item()
        )

    def attention_hook(_module, inputs):
        weights = inputs[0].detach()[0, 0, :16, :16].float().cpu()
        observed["attention"] = {
            "layer": 0,
            "head": 0,
            "tokens": [tokenizer.decode([i]) for i in input_ids[0, :16].tolist()],
            "weights": [[finite(value) for value in row] for row in weights.tolist()],
        }

    stages = [("embeddings", model.embeddings)]
    stages += [(f"blocks.{i}", block) for i, block in enumerate(model.blocks)]
    stages += [("final_norm", model.final_norm), ("lm_head", model.lm_head)]
    try:
        for name, module in stages:
            handles.append(module.register_forward_pre_hook(start_module(name)))
            handles.append(module.register_forward_hook(activation_hook(name)))
        handles.append(model.lm_head.register_forward_hook(logits_hook))
        if model.blocks:
            handles.append(model.blocks[0].attention.attention_dropout.register_forward_pre_hook(attention_hook))
        loss = train_step(model, optimizer, input_ids, targets, timings=timings)
    finally:
        for handle in handles:
            handle.remove()

    sums = defaultdict(lambda: {"gradient": 0.0, "parameter": 0.0, "update": 0.0, "count": 0})
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
        value = parameter.detach().float()
        sums[group]["parameter"] += value.square().sum().item()
        sums[group]["update"] += (value - before[name].float()).square().sum().item()
        if parameter.grad is not None:
            sums[group]["gradient"] += parameter.grad.detach().float().square().sum().item()
        sums[group]["count"] += parameter.numel()

    layers = []
    for name, _module in stages:
        layer = snapshots[name]
        layer["parameter_count"] = sums[name]["count"]
        for key in ("gradient", "parameter", "update"):
            layer[key + "_norm"] = finite(math.sqrt(sums[name][key]))
        layers.append(layer)

    return loss, {
        "version": 1,
        "input_ids": input_ids[0, :16].tolist(),
        "target_ids": targets[0, :16].tolist(),
        "input_shape": list(input_ids.shape),
        "pipeline": {
            "embeddings": module_ms["embeddings"],
            "transformer": sum(value for name, value in module_ms.items() if name.startswith("blocks.")) + module_ms["final_norm"],
            "prediction": module_ms["lm_head"] + timings["loss"],
            "backward": timings["backward"],
            "optimizer": timings["optimizer"],
        },
        "layers": layers,
        "gradient_norm": finite(math.sqrt(sum(item["gradient"] for item in sums.values()))),
        "parameter_norm": finite(math.sqrt(sum(item["parameter"] for item in sums.values()))),
        "update_norm": finite(math.sqrt(sum(item["update"] for item in sums.values()))),
        **observed,
    }
