import torch
import torch.nn.functional as F
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
    model.train()

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    model.zero_grad(set_to_none=True)

    logits = model(input_ids)

    batch_size, sequence_length, vocab_size = (
        logits.shape
    )

    loss = F.cross_entropy(
        logits.reshape(
            batch_size* sequence_length,
            vocab_size,
        ),
        target_ids.reshape(
            batch_size * sequence_length
        ),
    )

    loss.backward()

    lm_head_grad = model.lm_head.weight.grad
    embedding_grad = (
        model.embeddings.token_embedding.weight.grad
    )

    print("device:", device)

    print("logits shape:", logits.shape)
    print("targets shape:", target_ids.shape)

    print(
        "flattened logits shape:",
        logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        ).shape,
    )

    print(
        "flattened targets shape:",
        target_ids.reshape(
            batch_size * sequence_length
        ).shape,
    )

    print("loss:", loss.item())

    print(
        "lm head gradient shape:",
        lm_head_grad.shape,
    )

    print(
        "lm head gradient finite:",
        torch.isfinite(lm_head_grad).all().item(),
    )

    print(
        "lm head gradient norm:",
        lm_head_grad.norm().item(),
    )

    print(
        "embedding gradient finite:",
        torch.isfinite(embedding_grad).all().item(),
    )

    print(
        "embedding gradient norm:",
        embedding_grad.norm().item(),
    )

if __name__ == "__main__":
    main()