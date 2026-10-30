
# SLM From Scratch

从零开始构建、训练和运行一个小语言模型。

目标不是调用现成的预训练模型，而是先使用 PyTorch 搭建一个可以训练和生成文本的 TinyGPT，然后逐步深入到底层算子、C++、CUDA 和 Triton。

## 双人协作

团队成员为 **DonZhong** 和 **DunhuaZhong**，两人均可提交代码和合并 PR。采用独立功能分支、交叉评审和共同联调。

- [协作指南](CONTRIBUTING.md)：权限、建议分工、分支与合并流程、检查命令。
- [双人开发需求与计划](docs/development-plan.md)：当前基线、训练监控闭环与推理功能的任务及验收条件。
- [平台说明](docs/platform-v1.md)：已有服务、API 与本地启动步骤。
- [训练工作台](docs/training-workspace.md)：训练过程回看、逐层诊断、注意力与预测概率，以及升级步骤。

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
- C++ Drogon APIs and PostgreSQL persistence for training runs, metrics, and hardware snapshots.
- Python training-metrics reporting and host hardware collection.
- React hardware dashboard and training-metrics panel.
- Training workspace with step replay, EMA loss, perplexity, epoch summaries and CSV export.
- Opt-in snapshots of layer activations, gradients, actual parameter updates, token predictions and causal attention.

Current limitations:

- TinyGPT is a small educational base language model.
- The current training objective is next-token prediction, not instruction following.
- Checkpoints have been tested for model restoration, but exact interruption/resumption of training is not yet implemented.
- Training monitoring has an implementation; complete local integration acceptance is tracked in the development plan.
- Web-based model chat and a dedicated inference service are planned, not yet implemented.

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
| `--api-base-url` | Optional monitoring backend URL |
| `--run-name` | Optional experiment name when reporting to the backend |
| `--diagnostics-interval` | Sample model internals every N steps; 0 disables sampling |

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

Report the results of the current test run; test counts change as coverage grows.

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


## Text Generation

The project supports autoregressive text generation from a trained TinyGPT checkpoint.

Run the current generation example:

```bash
python scripts/generate_check.py
```

The script currently loads:

`/mnt/d/SLM/checkpoints/tinygpt-tinyshakespeare-full-20261027.pt`

Current generation settings:

- Prompt: `ROMEO:`
- Maximum new tokens: 300
- Temperature: 0.8
- Top-k: 10
- Sampling: multinomial sampling from the filtered probability distribution
- Unknown token: excluded from generation

The tokenizer is character-level, so one token corresponds to one character rather than one word or subword.

Generation is stochastic, so different runs may produce different text.

The current model is an experimental learning baseline. Successful inference does not imply fluent or semantically coherent generation.

Text generation loads the existing checkpoint and does not retrain the model or save a new checkpoint.

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

当前已实现 PostgreSQL 实验与指标存储、硬件快照 API、Python 指标上报，以及 React 硬件和训练监控页面。TinyGPT Chat 尚未接入。

首次启动需要配置 `.env` 并按顺序执行数据库迁移；完整步骤见 [平台说明](docs/platform-v1.md)。上面的健康检查命令用于已完成初始化的环境。
