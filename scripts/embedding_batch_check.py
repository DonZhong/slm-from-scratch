import torch
from torch.utils.data import DataLoader

from slm.config import TinyGPTConfig
from slm.data import NextTokenDataset
from slm.model import TinyGPTEmbeddings
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

    model = TinyGPTEmbeddings(config).to(device)

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    hidden_states = model(input_ids)

    print("device:", device)

    print("input shape:", input_ids.shape)

    print("target shape:", target_ids.shape)

    print("hidden states shape:", hidden_states.shape)
    print("hidden states dtype:", hidden_states.dtype)
    print("hidden states device:", hidden_states.device)

    print(
        "token embedding weight shape:",
        model.token_embedding.weight.shape,
    )

    print(
        "position embedding weight shape:",
        model.position_embedding.weight.shape,
    )

if __name__ == "__main__":
    main()