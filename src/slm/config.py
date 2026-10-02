from dataclasses import dataclass


@dataclass
class TinyGPTConfig:
    vocab_size:int

    context_length: int = 128
    d_model: int = 128

    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512

    dropout: float = 0.1

    def __post_init_(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 9")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisivle b n_heads"
            )

@property
def head_dim(self) -> int:
    return self.d_model // self.n_heads