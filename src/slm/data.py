import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ):
        if context_length <= 0:
            raise ValueError(
                "context_length must be greater than 0"
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

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = index + self.context_length

        input_ids = self.tokens[start:end]

        target_ids = self.tokens[
            start + 1 : end + 1
        ]

        return input_ids, target_ids