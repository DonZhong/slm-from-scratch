import math

import torch
from torch import nn

from .config import TinyGPTConfig

class TinyGPTEmbeddings(nn.Module):
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

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        token_vectors = self.token_embedding(input_ids)

        position_vectors = self.position_embedding(positions)
        position_vectors = position_vectors.unsqueeze(0)

        hidden_states = token_vectors + position_vectors
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        self.attention_dropout = nn.Dropout(
            config.dropout
        )

        mask = torch.tril(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, channels = (
            hidden_states.shape
        )

        if channels != self.d_model:
            raise ValueError(
                "hidden_states last dimension"
                "must equal d_model"
            )

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = (
            query @ key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        context = attention_weights @ value

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )

        output = self.out_proj(context)

        return output

class FeedForward(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.fc_in = nn.Linear(
            config.d_model,
            config.d_ff,
        )

        self.activation = nn.GELU()

        self.fc_out = nn.Linear(
            config.d_ff,
            config.d_model,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.norm2 = nn.LayerNorm(
            config.d_model
        )

        self.feed_forward = FeedForward(
            config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.norm1(hidden_states)
        )

        hidden_states = hidden_states + self.feed_forward(
            self.norm2(hidden_states)
        )

        return hidden_states

class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.config = config

        self.embeddings = TinyGPTEmbeddings(config)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(
            hidden_states
        )

        logits = self.lm_head(
            hidden_states
        )

        return logits