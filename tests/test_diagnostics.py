import copy
import json

import pytest
import torch

from slm.config import TinyGPTConfig
from slm.diagnostics import diagnostic_train_step
from slm.model import TinyGPT
from slm.tokenizer import CharacterTokenizer
from slm.training import train_step


def setup_step():
    torch.manual_seed(7)
    tokenizer = CharacterTokenizer.from_text("abcde")
    model = TinyGPT(TinyGPTConfig(
        vocab_size=tokenizer.vocab_size, context_length=4,
        d_model=8, n_heads=2, n_layers=2, d_ff=16, dropout=0,
    ))
    inputs = torch.tensor([tokenizer.encode("abcd")])
    targets = torch.tensor([tokenizer.encode("bcde")])
    return model, tokenizer, inputs, targets


def test_diagnostics_observe_actual_step_without_changing_training():
    model, tokenizer, inputs, targets = setup_step()
    baseline = copy.deepcopy(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
    expected_loss = train_step(baseline, baseline_optimizer, inputs, targets)
    loss, diagnostics = diagnostic_train_step(model, optimizer, inputs, targets, tokenizer)
    assert loss.item() == pytest.approx(expected_loss.item())
    for actual, expected in zip(model.parameters(), baseline.parameters()):
        assert torch.equal(actual, expected)
    expected_update = sum((p.detach() - before[name]).square().sum().item() for name, p in model.named_parameters()) ** 0.5
    assert diagnostics["update_norm"] == pytest.approx(expected_update, rel=1e-5)
    assert diagnostics["gradient_norm"] > 0
    assert 0 <= diagnostics["token_accuracy"] <= 1
    assert len(diagnostics["layers"]) == 5
    assert diagnostics["prediction"]["context"] == "abcd"
    assert diagnostics["prediction"]["target"] == "e"
    assert sum(t["probability"] for t in diagnostics["prediction"]["top_tokens"]) <= 1.000001
    attention = torch.tensor(diagnostics["attention"]["weights"])
    assert torch.equal(attention.triu(1), torch.zeros_like(attention))
    assert torch.allclose(attention.sum(-1), torch.ones(4), atol=1e-6)
    json.dumps(diagnostics, allow_nan=False)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_diagnostic_hooks_are_removed_when_training_fails(monkeypatch):
    model, tokenizer, inputs, targets = setup_step()
    optimizer = torch.optim.AdamW(model.parameters())
    def fail():
        raise RuntimeError("optimizer failed")
    monkeypatch.setattr(optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        diagnostic_train_step(model, optimizer, inputs, targets, tokenizer)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
