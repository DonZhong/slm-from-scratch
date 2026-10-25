
# SLM From Scratch

从零开始构建、训练和运行一个小语言模型。

目标不是调用现成的预训练模型，而是先使用 PyTorch 搭建一个可以训练和生成文本的 TinyGPT，然后逐步深入到底层算子、C++、CUDA 和 Triton。

## Goals

### 1. TinyGPT

第一阶段完成一个 decoder-only Transformer：

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

```text
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
```

---


## Current Progress

SLM From Scratch is an educational project for building and understanding small language models from the ground up.

Implemented:

- Character-level tokenizer with unknown-token handling.
- Sliding-window next-token dataset.
- TinyGPT Transformer with causal self-attention.
- PyTorch training with AdamW and cross-entropy loss.
- Train/validation split and validation loss evaluation.
- Training CLI with configurable epochs, batch size, learning rate, and maximum steps.
- Checkpoint saving and model weight restoration.
- Automated unit and CLI integration tests.

Current limitations:

- TinyGPT is a small educational base language model.
- The current training objective is next-token prediction, not instruction following.
- Checkpoints have been tested for model restoration, but exact interruption/resumption of training is not yet implemented.
- Web-based model chat and training monitoring are planned, not yet implemented.

## Installation

Development environment:

- Windows with WSL2 Ubuntu
- Python 3.12
- PyTorch with CUDA support
- NVIDIA GPU

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install a compatible PyTorch build for your environment, then install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Training

The training entry point is:

```text
scripts/train.py
```

Example using a local text dataset:

```bash
python scripts/train.py \
  --data-path /mnt/d/SLM/datasets/slm-from-scratch/train-smoke.txt \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 0.0005 \
  --max-steps 10
```

CLI parameters:

| Parameter | Description |
| --- | --- |
| `--data-path` | Path to a UTF-8 text dataset |
| `--epochs` | Maximum number of training epochs |
| `--batch-size` | Number of samples per training batch |
| `--learning-rate` | AdamW learning rate |
| `--max-steps` | Optional limit on optimizer updates |
| `--checkpoint-path` | Optional checkpoint output path |

The training pipeline splits text into training and validation datasets before constructing token windows.

The tokenizer vocabulary is built from the training text.

## Testing

Run the automated test suite:

```bash
python -m pytest tests/ -v
```

The tests cover:

- Text splitting and input validation.
- Language-model loss calculations.
- Model parameter updates.
- Evaluation without parameter updates.
- Training CLI step limits.
- Checkpoint saving, loading, and model inference.

The latest validated result is 12 passing tests.

## Checkpoints

Checkpoints are saved only when `--checkpoint-path` is provided.

Example:

```bash
python scripts/train.py \
  --epochs 3 \
  --batch-size 8 \
  --max-steps 10 \
  --checkpoint-path /mnt/d/SLM/checkpoints/slm-from-scratch/tinygpt-example.pt
```

A checkpoint contains:

- Model state dictionary.
- Optimizer state dictionary.
- Model configuration.
- Tokenizer vocabulary mapping.
- Epoch and global step.
- Training and validation losses.

Existing checkpoint files are not overwritten.

Large datasets and checkpoints should remain outside the Git repository, preferably on the D: drive.

## Roadmap

### Training Platform

- C++ Drogon backend.
- PostgreSQL database.
- React + TypeScript frontend.
- Docker Compose orchestration.
- Real-time training metrics and experiment tracking.
- GPU, CPU, and memory monitoring.

### TinyGPT Chat

- Load locally trained TinyGPT checkpoints.
- Run inference using Python and PyTorch.
- Build a web-based model chat interface.
- Stream generated tokens to the frontend.
- Support model selection and generation parameters.

### Model Development

- Train on larger English datasets.
- Prepare Chinese text datasets.
- Explore instruction tuning and conversational training.
- Study custom tensor operations, C++ extensions, CUDA kernels, and Triton.



## Training Monitor Platform V1

平台后端使用 C++17、Drogon 和 Docker Compose。

### 启动后端

```bash
docker compose build backend
docker compose up -d backend
```

### 验证接口

```bash
curl -i http://127.0.0.1:8080/api/health
python scripts/check_backend_health.py
```

预期健康接口返回：

```json
{
  "service": "slm-platform-backend",
  "status": "ok"
}
```

### 查看日志

```bash
docker compose logs --tail=50 backend
```

### 停止服务

```bash
docker compose down
```

当前已实现后端健康检查。PostgreSQL、React 前端、
训练监控和 TinyGPT Chat 尚未接入。
