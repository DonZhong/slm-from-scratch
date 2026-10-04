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

    print("text:", text)
    print("token ids:", token_ids)

    print("dataset size:", len(dataset))

    input_ids, target_ids = dataset[0]

    print(
        "input text:",
        tokenizer.decode(input_ids.tolist()),
    )

    print(
        "target text:",
        tokenizer.decode(target_ids.tolist()),
    )

if __name__ == "__main__":
    main()