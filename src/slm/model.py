import torch
from torch import nn

from .config import TinyGPTConfig

class TinyGPTEmbeddings(nn.module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.config =config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must shape [batch_size, sequenc_length]"
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.context_length:
            raise ValueError(
                "sequence_length cannot exceed context_length"
            )

        position = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        token_vectors = self.token_embedding(input_ids)

        position_vectors = self.position_embedding(positions)
        position_vectors = position_vectors.unsqueeze(0)

        hidden_states = token_vectors + position_vectors
        hidden_states = self.dropout(hidden_states)

        return hidden_states

        