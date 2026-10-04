from torch.utils.data import DataLoader

from slm.data import NextTokenDataset
from slm.tokenizer import CharacterTokenizer

def main():
    text = "hello tiny gpt"

    tokenizer = CharacterTokenizer.from_text(text)

    token_ids = tokenizer.encode(text)

    dataset = NextTokenDataset(
        token_ids=token_ids,
        context_length=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
    )

    input_batch, target_batch = next(iter(loader))

    print("input batch:")
    print(input_batch)

    print("target batch:")
    print(target_batch)

    print("input shape:", input_batch.shape)
    print("target shape:", target_batch.shape)

    for row in range(input_batch.shape[0]):
        input_text = tokenizer.decode(
            input_batch[row].tolist()
        )

        target_text = tokenizer.decode(
            target_batch[row].tolist()
        )

        print(
            f"sample {row}:"
            f"{input_text!r} -> {target_text!r}"
        )

if __name__ == "__main__":
    main()