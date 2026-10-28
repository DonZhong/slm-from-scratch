from pathlib import Path
import math
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
    temperature: float = 0.8,
    top_k: int | None = 10,
) -> str:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "temperature must be finite and positive"
        )

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("prompt cannot be empty")

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

            logits = model(input_ids)[:, -1, :].squeeze(0)

            # 1. 调整概率分布的尖锐程度
            logits = logits / temperature

            # 2. 不生成训练中未使用的 <unk> 字符
            logits[tokenizer.unk_id] = float("-inf")

            # 3. 只保留得分最高的 K 个候选字符
            if top_k is not None:
                # 排除 <unk>，避免生成未知字符
                candidate_indices = torch.arange(
                    logits.size(-1),
                    device=logits.device,
                )
                candidate_indices = candidate_indices[
                    candidate_indices != tokenizer.unk_id
                ]

                # K 不能超过实际可用字符数量
                k = min(top_k, candidate_indices.numel())

                # 只在合法字符中选取得分最高的 K 个
                candidate_logits = logits[candidate_indices]
                selected_positions = torch.topk(
                    candidate_logits,
                    k,
                ).indices
                selected_indices = candidate_indices[selected_positions]

                # 其他字符全部屏蔽，保证候选集合大小受控
                filtered_logits = torch.full_like(
                    logits,
                    float("-inf"),
                )
                filtered_logits[selected_indices] = logits[selected_indices]
                logits = filtered_logits

            # 4. 将得分转换为概率，再随机采样
            probabilities = torch.softmax(logits, dim=-1)

            next_token_id = torch.multinomial(
                probabilities,
                num_samples=1,
            ).item()

            token_ids.append(next_token_id)

    return tokenizer.decode(token_ids)


def main():
    checkpoint_path = Path(
        "/mnt/d/SLM/checkpoints/"
        "tinygpt-tinyshakespeare-full-20261027.pt"
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

    prompt = "ROMEO:"

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=300,
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