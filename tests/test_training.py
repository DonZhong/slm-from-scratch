import pytest
import torch
from torch import nn
from torch.optim import AdamW
import math

from slm.training import evaluate, language_model_loss, train_step

from torch.utils.data import DataLoader, TensorDataset

class TinyTestModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.lm_head(hidden)


def test_language_model_loss_returns_finite_scalar() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0],
                [0.5, 1.5, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [[0, 1]],
        dtype=torch.long,
    )

    loss = language_model_loss(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_train_step_updates_model_parameters() -> None:
    torch.manual_seed(42)

    model = TinyTestModel(
        vocab_size=5,
        hidden_size=8,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-2,
    )

    input_ids = torch.tensor(
        [[0, 1, 2]],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    weight_before = model.lm_head.weight.detach().clone()

    loss = train_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
    )

    weight_after = model.lm_head.weight.detach().clone()

    assert torch.isfinite(loss)
    assert not torch.equal(weight_before, weight_after)


@pytest.mark.parametrize(
    ("logits_shape", "targets_shape", "error_message"),
    [
        ((2, 3), (2,), "logits must have shape"),
        ((1, 2, 3), (1, 2, 1), "targets must have shape"),
        ((1, 2, 3), (1, 3), "logits and targets must agree"),
    ],
)
def test_language_model_loss_rejects_invalid_shapes(
    logits_shape: tuple[int, ...],
    targets_shape: tuple[int, ...],
    error_message: str,
) -> None:
    logits = torch.zeros(logits_shape)
    targets = torch.zeros(targets_shape, dtype=torch.long)

    with pytest.raises(ValueError, match=error_message):
        language_model_loss(logits, targets)


def test_language_model_loss_matches_manual_calculation() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0],
                [math.log(3.0), 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [[1, 0]],
        dtype=torch.long,
    )

    expected_loss = (
        math.log(2.0) + math.log(4.0 / 3.0)
    ) / 2.0

    actual_loss = language_model_loss(logits, targets)

    assert actual_loss.item() == pytest.approx(
        expected_loss,
        rel=1e-6,
    )

def test_evaluate_returns_expected_loss_without_updating_weights() -> None:
    torch.manual_seed(42)

    model = TinyTestModel(
        vocab_size=5,
        hidden_size=8,
    )

    input_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3]],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [[1, 2], [2, 3], [3, 4]],
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(input_ids, targets),
        batch_size=2,
        shuffle=False,
    )

    weight_before = model.lm_head.weight.detach().clone()

    with torch.inference_mode():
        expected_loss = language_model_loss(
            model(input_ids),
            targets,
        ).item()

    actual_loss = evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert actual_loss == pytest.approx(
        expected_loss,
        rel=1e-6,
    )

    assert torch.equal(
        weight_before,
        model.lm_head.weight,
    )

    assert model.training

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )
