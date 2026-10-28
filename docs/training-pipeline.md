
# TinyGPT Training Pipeline

## 1. Purpose

本文记录 SLM From Scratch 项目当前实现的 TinyGPT 训练流程、核心模块、模型评估以及 Checkpoint 机制。

当前模型是用于学习语言模型原理的 decoder-only Transformer。

训练目标是 Next Token Prediction（下一个 Token 预测），尚不是经过指令微调的对话模型。

## 2. Pipeline Overview

```text
Raw UTF-8 Text
       |
       v
split_text()
       |
       +------ Train Text (90%)
       |           |
       |           v
       |     CharacterTokenizer
       |           |
       |           v
       |     Train Dataset
       |           |
       |           v
       |       DataLoader
       |           |
       |           v
       |      train_step()
       |           |
       |           v
       |      Model Update
       |
       +------ Validation Text (10%)
                   |
                   v
             Same Tokenizer
                   |
                   v
             Validation Dataset
                   |
                   v
                evaluate()
                   |
                   v
             Validation Loss

             Model + Metadata
                   |
                   v
              Checkpoint
```

## 3. Core Modules

### src/slm/data.py

负责：

- 构造滑动窗口训练样本
- 划分训练集和验证集
- 检查上下文长度和数据有效性

### src/slm/training.py

负责：

- 计算语言模型交叉熵损失
- 执行单个训练 Step
- 计算验证集 Loss

### scripts/train.py

负责：

- 解析 CLI 参数
- 读取原始文本
- 初始化 Tokenizer、模型和优化器
- 执行 Epoch 和 Batch 循环
- 统计训练与验证损失
- 保存 Checkpoint

## 4. Training Objective

给定输入序列：

x = [x1, x2, ..., xT]

训练目标是预测下一个 Token：

y = [x2, x3, ..., x(T+1)]

使用 Cross Entropy Loss：

L = -(1/N) * sum(log(p(correct_token)))

其中：

- N：参与计算的目标 Token 数量
- p(correct_token)：模型给正确目标 Token 的预测概率
- L：平均负对数似然损失

Loss 越低，表示模型在对应数据上的平均预测负对数似然越小。

## 5. Train / Validation Split

当前训练脚本按文本顺序进行：

- Training：90%
- Validation：10%

先切分原始文本，再构造滑动窗口样本。

这样可以避免训练集和验证集窗口跨越切分边界。

Tokenizer 词表使用训练文本建立。

验证集出现训练词表之外的字符时，使用 `<unk>` 映射。

需要注意：当前验证集是连续文本尾部切分，不代表完全独立的外部测试集。

## 6. Training Step

每次训练执行：

1. `model.train()`
2. `optimizer.zero_grad()`
3. Forward
4. Cross Entropy Loss
5. Backward
6. `optimizer.step()`

训练阶段会更新模型参数。

## 7. Validation

验证阶段执行：

1. `model.eval()`
2. `torch.inference_mode()`
3. Forward
4. 计算 Validation Loss
5. 恢复原来的模型模式

验证阶段不执行 Backward 和优化器更新。

当前 Train Loss 是训练过程中多个 Step 的加权平均。

Validation Loss 则是在评估时使用当前模型参数计算。

因此二者不属于完全相同条件下的评估指标。

## 8. Checkpoint

当前 Checkpoint 包含：

- Model state_dict
- Optimizer state_dict
- Model configuration
- Tokenizer vocabulary mapping
- Epoch
- Global step
- Training Loss
- Validation Loss

Checkpoint 使用 `torch.save()` 保存。

已通过自动化测试验证：

训练 → 保存 → 读取 → 重建模型 → 加载权重 → Forward。

当前尚未实现严格可复现的断点续训。

## 9. Validation Commands

运行完整自动化测试：

```bash
python -m pytest tests/ -v
```

运行训练 Smoke Test：

```bash
python scripts/train.py \
  --data-path /mnt/d/SLM/datasets/slm-from-scratch/train-smoke.txt \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 0.0005 \
  --max-steps 10
```

历史验证覆盖 CUDA 训练、Train/Validation Loss 及 Checkpoint roundtrip。当前提交应重新执行所需检查并记录实际结果，不沿用历史测试数量。

## 10. Future Development

已有 C++ Drogon 监控后端、PostgreSQL 实验数据库、React 监控页面、Docker Compose 编排及训练指标上报实现。

下一阶段先完成训练异常状态处理和端到端监控验收，再实现本地 TinyGPT 推理服务及网页文本生成。双人分工和验收条件见 [开发计划](development-plan.md)。

模型训练和推理继续使用 Python / PyTorch。

C++ 后端负责平台业务、API 和服务通信。

## 11. Training Monitor API

### 11.1 Architecture

训练过程由 Python + PyTorch 执行，C++ Drogon 后端负责实验管理和指标写入，PostgreSQL 负责持久化实验与训练指标。

```text
PyTorch Training Script
        |
        | HTTP + JSON
        v
C++ Drogon Backend
        |
        v
PostgreSQL
        ^
        |
React + TypeScript Monitor
```

### 11.2 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | 检查后端健康状态 |
| GET | `/api/db/health` | 检查数据库连接 |
| POST | `/api/training-runs` | 创建训练实验 |
| GET | `/api/training-runs` | 查询实验列表 |
| POST | `/api/training-runs/{run_id}/status` | 更新实验状态 |
| POST | `/api/training-runs/{run_id}/metrics` | 写入训练指标 |
| GET | `/api/training-runs/{run_id}/metrics` | 查询实验指标 |

创建实验时，`name` 和 `dataset_name` 为必填字段，`model_name` 可选，默认值为 `TinyGPT`。

训练指标包含：

- `global_step`：全局训练步数
- `epoch`：当前训练轮次
- `train_loss`：当前 Step 的训练损失
- `val_loss`：验证损失，可为空
- `learning_rate`：当前学习率
- `tokens_per_second`：每秒处理的目标 Token 数，可为空

### 11.3 Start a Monitored Training Run

```bash
python scripts/train.py \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 0.0005 \
  --max-steps 5 \
  --api-base-url http://127.0.0.1:8080 \
  --run-name pytorch-metrics-smoke-test
```

不提供 `--api-base-url` 时，训练脚本不创建后端实验，也不向后端写入训练指标。

正常完成训练后，脚本将实验状态更新为 `completed`。

### 11.4 Checkpoint Policy

Checkpoint 默认不保存。

只有显式指定 `--checkpoint-path` 时，脚本才写入模型、优化器状态及相关训练元数据。

提交代码、更新文档与保存模型 Checkpoint 是三件不同的事。

### 11.5 Validation

```bash
python -m py_compile scripts/train.py
python -m pytest tests/ -q
git diff --check
```

后端修改后，还应重新构建 Docker 镜像并验证 HTTP API。

自动化测试结果以实际执行输出为准。
