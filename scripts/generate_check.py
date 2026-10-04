from pathlib import Path

import torch

from slm.config import TinyGPTConfig
from slm.model import TinyGPT
from slm.tokenizer import CharacterTokenizer


def restore_tokenizer(
    token_to_id: dict[str, int],
) -> CharacterTokenizer:
    ordered_items = sorted(
        token_to_id.items(),
        key=lambda item: item[1],
    )

    ordered_tokens = [
        token
        for token, _ in ordered_items
    ]

    if not ordered_tokens:
        raise ValueError(
            "checkpoint tokenizer is empty"
        )

    if ordered_tokens[0] != "<unk>":
        raise ValueError(
            "checkpoint token 0 must be <unk>"
        )

    tokenizer = CharacterTokenizer(
        vocabulary=ordered_tokens[1:],
    )

    if tokenizer.token_to_id != token_to_id:
        raise ValueError(
            "restored tokenizer does not match checkpoint"
        )

    return tokenizer


def generate(
    model: TinyGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError(
            "prompt cannot be empty"
        )

    model.eval()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context_ids = token_ids[
                -model.config.context_length:
            ]

            input_ids = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=device,
            )

            logits = model(input_ids)

            next_token_logits = logits[
                :,
                -1,
                :,
            ]

            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1,
            ).item()

            token_ids.append(
                next_token_id
            )

    return tokenizer.decode(token_ids)


def main():
    checkpoint_path = Path(
        "/mnt/d/SLM/checkpoints/"
        "slm-from-scratch/"
        "tinygpt-toy-v1.pt"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    config = TinyGPTConfig(
        **checkpoint["config"]
    )

    tokenizer = restore_tokenizer(
        checkpoint["token_to_id"]
    )

    model = TinyGPT(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    prompt = "hel"

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=40,
        device=device,
    )

    print("device:", device)
    print("checkpoint:", checkpoint_path)
    print("checkpoint epoch:", checkpoint["epoch"])
    print("checkpoint step:", checkpoint["global_step"])
    print("checkpoint loss:", checkpoint["loss"])
    print("vocab size:", tokenizer.vocab_size)

    print()
    print("prompt:")
    print(repr(prompt))

    print()
    print("generated:")
    print(repr(generated_text))

    print()
    print("generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()