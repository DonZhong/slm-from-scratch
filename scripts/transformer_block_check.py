import torch
from torch.utils.data import DataLoader

from slm.config import TinyGPTConfig
from slm.data import NextTokenDataset
from slm.model import (
    TinyGPTEmbeddings,
    TransformerBlock,
)
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

    input_ids, _ = next(iter(loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    embeddings = TinyGPTEmbeddings(config).to(device)
    block = TransformerBlock(config).to(device)

    embeddings.eval()
    block.eval()

    input_ids = input_ids.to(device)

    with torch.no_grad():
        hidden_states = embeddings(input_ids)
        output = block(hidden_states)

    print("device:", device)
    print("input ids shape:", input_ids.shape)
    print("embedding shape:", hidden_states.shape)
    print("block output shape:", output.shape)

    print(
        "feed-forward input weight shape:",
        block.feed_forward.fc_in.weight.shape,
    )

    print(
        "feed-forward output weight shape:",
        block.feed_forward.fc_out.weight.shape,
    )

    print(
        "norm1 weight shape:",
        block.norm1.weight.shape,
    )

    print(
        "output finite:",
        torch.isfinite(output).all().item(),
    )

if __name__ == "__main__":
    main()