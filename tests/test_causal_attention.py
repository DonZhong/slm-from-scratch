
import torch

from slm.config import TinyGPTConfig
from slm.model import TinyGPT


def test_future_token_does_not_change_past_logits():
    torch.manual_seed(42)

    config = TinyGPTConfig(
        vocab_size=16,
        context_length=8,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
    )

    model = TinyGPT(config)
    model.eval()

    # 两组序列前四个 token 完全相同
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 4, 9]])

    # 只有最后一个 token 不同
    with torch.inference_mode():
        first_logits = model(first)
        second_logits = model(second)

    # 改变未来 token，不应该影响前四个位置的预测
    torch.testing.assert_close(
        first_logits[:, :4, :],
        second_logits[:, :4, :],
        rtol=1e-6,
        atol=1e-6,
    )
