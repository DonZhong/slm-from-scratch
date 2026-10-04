from slm.tokenizer import CharacterTokenizer

def main():
    training_text = "hello tiny gpt"

    tokenizer = CharacterTokenizer.from_text(
        training_text
    )

    sample_text = "hello"

    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)

    unknown_text = "hello!"
    unknown_ids = tokenizer.encode(unknown_text)

    print("vocab size:", tokenizer.vocab_size)
    print("token to id:", tokenizer.token_to_id)

    print("sample text:", sample_text)

    print("encoded:", token_ids)
    print("decoded:", decoded_text)

    print("unknown example:", unknown_ids)

    print(
        "round trip correct:",
        decoded_text == sample_text,
    )

if __name__ == "__main__":
    main()