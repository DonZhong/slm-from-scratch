
# Training Monitor Platform V1

## 1. 项目目标

在现有 TinyGPT 训练与推理代码的基础上，
逐步建设训练监控和模型聊天平台。

## 2. 技术栈

Python + PyTorch：TinyGPT 训练与评估

C++17 + Drogon：HTTP API 与实验管理

PostgreSQL：训练实验及指标持久化

React + TypeScript + Vite：Web 监控前端

Docker Compose：本地服务编排

## 3. 已实现功能

后端健康检查与数据库健康检查。

创建训练实验并查询实验列表。

更新训练实验状态。

写入和查询训练指标。

Python 训练脚本通过 HTTP API 接入实验管理。

采集训练损失、验证损失、学习率及 Token 吞吐量。

前端 Vite 将 /api 请求代理至 Docker 网络中的 backend:8080。

Python 硬件采集器上报 CPU、内存、磁盘、网络和可用的 GPU 信息，后端保存与查询硬件快照。

React 页面展示硬件状态和训练实验指标，支持实验切换与定时刷新。

以上为已有源码能力，完整联调验收见 [双人开发计划](development-plan.md)。TinyGPT 网页文本生成仍在规划中。

## 4. 本地初始化与验证

建议在 WSL2 Ubuntu 的项目目录操作。前提是已安装 Docker Engine / Compose，或已在 Docker Desktop 中启用此 Ubuntu 的 WSL integration。Python 训练环境按 README 安装 PyTorch 和项目依赖。

### 4.1 环境变量

仅首次创建本地配置时复制，不覆盖已有 `.env`：

```bash
cd ~/projects/slm-from-scratch
test -f .env || cp .env.example .env
# 编辑 .env，将 SLM_DB_PASSWORD=CHANGE_ME 改为自己的开发密码
docker compose config --quiet
```

### 4.2 新数据库初始化

首次创建空数据库时，先启动 PostgreSQL，再按顺序应用迁移。Compose 当前不会自动执行这些迁移。

```bash
docker compose up -d --wait postgres
docker compose exec -T postgres sh -c \
  'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < platform/database/migrations/001_initial_schema.sql
docker compose exec -T postgres sh -c \
  'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < platform/database/migrations/002_hardware_snapshots.sql
docker compose exec -T postgres sh -c \
  'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < platform/database/migrations/003_training_diagnostics.sql
```

已有数据库需先核对已执行迁移，不重复初始化或删除数据卷。后续新增迁移的编号由两人事先协调。

### 4.3 启动平台

```bash
docker compose up -d --build backend frontend
docker compose ps
curl -i http://127.0.0.1:8080/api/health
curl -i http://127.0.0.1:8080/api/db/health
curl -i http://127.0.0.1:5173/api/health
```

浏览器访问 `http://127.0.0.1:5173`。Vite 代理默认指向 Docker 服务名 `backend`，上述前端启动方式需要 Compose 网络；宿主机单独运行前端时，可在 `platform/frontend` 下执行 `SLM_API_PROXY_TARGET=http://127.0.0.1:8080 npm run dev`。

宿主机 8080 或 5173 被占用会导致启动失败，先检查端口使用情况，再协调本地端口映射及 API 调用地址。

### 4.4 上报硬件与训练指标

在已激活并安装项目依赖的 Python 环境执行一次硬件上报：

```bash
python -m pip install -e ".[dev,monitor]"
python scripts/hardware_monitor.py --once
curl -sS http://127.0.0.1:8080/api/hardware-snapshots/latest
```

持续采集可使用 `python scripts/hardware_monitor.py --interval 5`，按 Ctrl+C 停止。

指定自己准备的 UTF-8 文本执行有限步训练：

```bash
python scripts/train.py \
  --data-path /mnt/d/SLM/datasets/slm-from-scratch/train-smoke.txt \
  --epochs 1 --batch-size 8 --max-steps 5 \
  --api-base-url http://127.0.0.1:8080 \
  --run-name two-person-smoke
curl -sS http://127.0.0.1:5173/api/training-runs
```

数据路径应替换为实际存在的文件。使用训练输出或列表返回的真实 run ID 查询 `/api/training-runs/{run_id}/metrics`，并在页面选择该实验核对指标和最终状态。

### 4.5 检查与停止

```bash
python -m pytest tests/ -q
docker compose logs --tail=50 backend frontend
docker compose down
```

每次提交的测试结果以该次实际执行记录为准。后端 API 冒烟检查和所需测试记录见 `.github/workflows/backend-ci.yml`。`docker compose down` 保留数据卷。

## 5. 已有接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 后端健康检查 |
| GET | `/api/db/health` | 数据库健康检查 |
| POST / GET | `/api/training-runs` | 创建 / 查询实验 |
| POST | `/api/training-runs/{run_id}/status` | 更新实验状态 |
| POST / GET | `/api/training-runs/{run_id}/metrics` | 写入 / 查询训练指标 |
| POST / GET | `/api/hardware-snapshots` | 写入 / 查询硬件快照 |
| GET | `/api/hardware-snapshots/latest` | 查询最新硬件快照 |

接口或数据库改动需要同步另一位成员，并在 PR 中提供请求、响应、错误情况和联调结果。两人提交和合并规则见 [协作指南](../CONTRIBUTING.md)。
