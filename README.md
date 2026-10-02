# SLM From Scratch

从零开始构建、训练和运行一个小语言模型。

目标不是调用现成的预训练模型，而是先使用PyTorch搭建一个可以训练和生成文本的TinyGPT，然后逐步深入到底层算子、C++、CUDA和Triton。

## Goals

### 1. TinyGPT

第一阶段完成一个 decoder-only Transformer:

- Character-level tokenizer
- Token embedding
- Position embedding
- Causal self-attention
- Multi-head attention
- Feed-forward network
- Residual connection
- Normalization
- Cross entropy loss
- AdamW optimizer
- Checkpoint
- Text generation

训练流程：
Raw Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
TinyGPT
    ↓
Loss
    ↓
Backpropagation
    ↓
Optimizer
    ↓
Checkpoint