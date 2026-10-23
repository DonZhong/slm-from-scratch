
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

最新已完成验证：

- 12 automated tests passed
- CUDA training succeeded
- Train Loss calculated
- Validation Loss calculated
- Checkpoint roundtrip succeeded

## 10. Future Development

下一阶段规划：

- C++ Drogon monitoring backend
- PostgreSQL experiment database
- React + TypeScript dashboard
- Docker Compose deployment
- Real-time training metrics
- Local TinyGPT inference service
- Web chat interface for the trained model

模型训练和推理继续使用 Python / PyTorch。

C++ 后端负责平台业务、API 和服务通信。
