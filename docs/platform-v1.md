
# Training Monitor Platform V1

## 1. 项目目标

在现有 TinyGPT 训练与推理代码的基础上，
逐步建设训练监控和模型聊天平台。

## 2.技术栈：

Python + PyTorch：TinyGPT 训练与评估

C++17 + Drogon：HTTP API 与实验管理

PostgreSQL：训练实验及指标持久化

React + TypeScript + Vite：Web 监控前端

Docker Compose：本地服务编排

## 3.已实现功能

后端健康检查与数据库健康检查。

创建训练实验并查询实验列表。

更新训练实验状态。

写入和查询训练指标。

Python 训练脚本通过 HTTP API 接入实验管理。

采集训练损失、验证损失、学习率及 Token 吞吐量。

前端 Vite 将 /api 请求代理至 Docker 网络中的 backend:8080。

## 4.本地验证

启动服务：

docker compose up -d --build backend frontend

检查服务状态：

docker compose ps

验证前端 API 代理：

curl -i http://127.0.0.1:5173/api/health

验证实验指标：

curl -sS \
  http://127.0.0.1:5173/api/training-runs/16/metrics \
  | python -m json.tool

注意：示例中的实验 ID 必须替换为数据库中实际存在的 ID。

运行自动化测试：

python -m pytest tests/ -q

每次提交的测试结果应以该次实际执行记录为准。
