
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.generate_check import generate
from slm.tokenizer import CharacterTokenizer


class FixedLogitModel(nn.Module):
    """用于测试生成策略的模拟模型，不需要训练。"""

    def __init__(
        self,
        vocab_size: int,
        unk_id: int,
        preferred_token_id: int | None = None,
    ):
        super().__init__()
        self.config = SimpleNamespace(context_length=8)
        self.vocab_size = vocab_size
        self.unk_id = unk_id
        self.preferred_token_id = preferred_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=input_ids.device,
        )

        # 故意让 <unk> 得分最高，验证生成时会屏蔽它。
        logits[..., self.unk_id] = 20.0

        if self.preferred_token_id is not None:
            logits[..., self.preferred_token_id] = 10.0

        return logits


@pytest.fixture
def tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer(vocabulary=["a", "b", "c"])


def test_generate_preserves_prompt_and_appends_tokens(tokenizer):
    model = FixedLogitModel(
        vocab_size=tokenizer.vocab_size,
        unk_id=tokenizer.unk_id,
        preferred_token_id=tokenizer.token_to_id["b"],
    )

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        max_new_tokens=3,
        device=torch.device("cpu"),
        temperature=0.8,
        top_k=1,
    )

    assert result == "abbb"


def test_top_k_keeps_exactly_k_candidates(tokenizer, monkeypatch):
    model = FixedLogitModel(
        vocab_size=tokenizer.vocab_size,
        unk_id=tokenizer.unk_id,
    )

    candidate_counts = []

    def fake_multinomial(probabilities, num_samples):
        candidate_counts.append(
            int(torch.count_nonzero(probabilities).item())
        )
        return torch.tensor(
            [tokenizer.token_to_id["a"]],
            device=probabilities.device,
        )

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        max_new_tokens=1,
        device=torch.device("cpu"),
        temperature=0.8,
        top_k=1,
    )

    assert result == "aa"
    assert candidate_counts == [1]


@pytest.mark.parametrize(
    "temperature",
    [0, -0.1, float("nan"), float("inf")],
)
def test_generate_rejects_invalid_temperature(tokenizer, temperature):
    model = FixedLogitModel(
        tokenizer.vocab_size,
        tokenizer.unk_id,
    )

    with pytest.raises(ValueError, match="temperature"):
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt="a",
            max_new_tokens=1,
            device=torch.device("cpu"),
            temperature=temperature,
            top_k=1,
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_generate_rejects_invalid_top_k(tokenizer, top_k):
    model = FixedLogitModel(
        tokenizer.vocab_size,
        tokenizer.unk_id,
    )

    with pytest.raises(ValueError, match="top_k"):
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt="a",
            max_new_tokens=1,
            device=torch.device("cpu"),
            temperature=0.8,
            top_k=top_k,
        )


def test_generate_rejects_empty_prompt(tokenizer):
    model = FixedLogitModel(
        tokenizer.vocab_size,
        tokenizer.unk_id,
    )

    with pytest.raises(ValueError, match="prompt"):
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt="",
            max_new_tokens=1,
            device=torch.device("cpu"),
        )
