
import torch
from torch.utils.data import Dataset



class NextTokenDataset(Dataset):
    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int = 1,
    ):
        if context_length <= 0:
            raise ValueError(
                "context_length must be greater than 0"
            )

        if stride <= 0:
            raise ValueError(
                "stride must be greater than 0"
            )

        if len(token_ids) <= context_length:
            raise ValueError(
                "token sequence must be longer than context_length"
            )

        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        available_starts = (
            len(self.tokens) - self.context_length - 1
        )

        return available_starts // self.stride + 1

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        end = start + self.context_length

        input_ids = self.tokens[start:end]

        target_ids = self.tokens[
            start + 1 : end + 1
        ]

        return input_ids, target_ids



def split_text(
    text: str,
    *,
    train_fraction: float,
    context_length: int,
) -> tuple[str, str]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be between 0 and 1"
        )

    if context_length <= 0:
        raise ValueError(
            "context_length must be positive"
        )

    split_index = int(len(text) * train_fraction)

    train_text = text[:split_index]
    val_text = text[split_index:]

    if (
        len(train_text) <= context_length
        or len(val_text) <= context_length
    ):
        raise ValueError(
            "Both splits must contain more characters "
            "than context_length"
        )

    return train_text, val_text
