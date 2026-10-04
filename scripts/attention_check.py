import torch
from torch.utils.data import DataLoader

from slm.config import TinyGPTConfig
from slm.data import NextTokenDataset
from slm.model import (
    CausalSelfAttention,
    TinyGPTEmbeddings,
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
    attention = CausalSelfAttention(config).to(device)

    input_ids = input_ids.to(device)

    hidden_states = embeddings(input_ids)
    attention_output = attention(hidden_states)

    print("device:", device)
    print("input ids shape:", input_ids.shape)
    print("hidden states shape:", hidden_states.shape)
    print("attention output shape:", attention_output.shape)

    print(
        "q weight shape:",
        attention.q_proj.weight.shape,
    )

    print(
        "causal mask shape:",
        attention.causal_mask.shape,
    )

    print(
        attention.causal_mask[:4, :4]
    )

if __name__ == "__main__":
    main()