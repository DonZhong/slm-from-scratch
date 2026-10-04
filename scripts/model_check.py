import torch
from torch.utils.data import DataLoader

from slm.config import TinyGPTConfig
from slm.data import NextTokenDataset
from slm.model import TinyGPT
from slm.tokenizer import CharacterTokenizer

def main():
    text = "hello tiny gpt"

    tokenizer = CharacterTokenizer.from_text(text)

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
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
    model.eval()

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("device:", device)

    print(
        "number of transformer blocks:",
        len(model.blocks),
    )

    print("input shape:", input_ids.shape)
    print("target shape:", target_ids.shape)
    print("logits shape:", logits.shape)

    print(
        "lm head weight shape:",
        model.lm_head.weight.shape,
    )

    print(
        "parameter count:",
        parameter_count,
    )

    print(
        "logits finite:",
        torch.isfinite(logits).all().item(),
    )

if __name__ == "__main__":
    main()