
# Training Monitor Platform V1

## 1. 项目目标

在现有 TinyGPT 训练与推理代码的基础上，
逐步建设训练监控和模型聊天平台。

技术栈：

- Python + PyTorch：模型训练与推理
- C++17 + Drogon：HTTP 后端
- PostgreSQL：实验与训练指标存储（待开发）
- React + TypeScript：Web 前端（待开发）
- Docker Compose：容器管理

## 2. 已实现功能

### C++ Drogon 后端

接口：

GET /api/health

监听端口：

8080

正常响应：

```json
{
  "service": "slm-platform-backend",
  "status": "ok"
}
```

## 3. 本地运行

构建镜像：

```bash
docker compose build backend
```

启动后端：

```bash
docker compose up -d backend
```

测试接口：

```bash
python scripts/check_backend_health.py
```

查看日志：

```bash
docker compose logs --tail=50 backend
```

停止服务：

```bash
docker compose down
```

## 4. 自动化测试

Python：

```bash
python -m pytest -q
```

C++ 后端：

- GitHub Actions 构建 Docker 镜像
- 启动 Drogon 容器
- 调用 HTTP 健康接口
- 验证 HTTP 200 与 JSON 响应

GitHub CI 结果以实际运行结果为准。

## 5. 后续计划

1. PostgreSQL 数据库连接
2. React + TypeScript 前端
3. 实时训练指标采集
4. 自训练 TinyGPT 模型推理与聊天
5. WebSocket 实时通信
```
