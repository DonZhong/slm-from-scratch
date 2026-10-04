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

    batch_size, sequence_length, vocab_size = logits.shape

    loss = F.cross_entropy(
        logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        ),
        target_ids.reshape(
            batch_size * sequence_length,
        ),
    )

    return loss


def main():
    torch.manual_seed(42)

    text = "hello tiny gpt"

    tokenizer = CharacterTokenizer.from_text(text)

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        dropout=0.0,
    )

    dataset = NextTokenDataset(
        token_ids=tokenizer.encode(text),
        context_length=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
    )

    input_ids, target_ids = next(iter(loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TinyGPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    model.train()

    optimizer.zero_grad(set_to_none=True)

    loss_before = compute_loss(
        model,
        input_ids,
        target_ids,
    )

    weight_before = (
        model.lm_head.weight
        .detach()
        .clone()
    )

    loss_before.backward()

    optimizer.step()

    weight_after = (
        model.lm_head.weight
        .detach()
        .clone()
    )

    model.eval()

    with torch.no_grad():
        loss_after = compute_loss(
            model,
            input_ids,
            target_ids,
        )

    weight_change = (
        weight_after - weight_before
    )

    print("device:", device)

    print(
        "loss before step:",
        loss_before.item(),
    )

    print(
        "loss after step:",
        loss_after.item(),
    )

    print(
        "lm head weight changed:",
        not torch.equal(
            weight_before,
            weight_after,
        ),
    )

    print(
        "weight change norm:",
        weight_change.norm().item(),
    )

    print(
        "weights finite:",
        torch.isfinite(weight_after).all().item(),
    )


if __name__ == "__main__":
    main()