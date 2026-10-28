# 双人协作开发

本项目由 **DonZhong** 和 **DunhuaZhong** 共同开发。两人都可以提交、推送功能分支、创建 PR 和合并 PR；合并职责不固定给某一个人。

## 权限与职责

| 成员 | GitHub 当前角色 | 开发权限 |
| --- | --- | --- |
| DonZhong | 仓库所有者 / Admin | 提交、推送、评审、合并及仓库设置 |
| DunhuaZhong | Collaborator / Write | 提交、推送、评审及合并 |

以上权限于 2026-10-28 核对。开发和合并不需要给协作者 Admin 权限。

建议初始分工如下，领取任务时可以互换：

- **DonZhong**：TinyGPT、数据、训练、评估及 Python 采集器；主要目录为 `src/slm/`、`scripts/`、`tests/`。
- **DunhuaZhong**：C++ API、数据库、React 界面和容器编排；主要目录为 `platform/`、`docker-compose.yml`。
- **共同负责**：需求、接口契约、CI、联调、文档和发布。跨模块任务先约定请求与响应，再分别实现。

模块负责人是默认协调人，不是唯一可以修改该模块的人。开发任务与验收条件见 [双人开发计划](docs/development-plan.md)。

## 一项任务，一个分支，一个 PR

1. 先记录任务目标、负责人、涉及模块、依赖和可验证的验收条件。另一位成员作为评审人。
2. 从最新 `main` 创建自己的分支，不共用正在开发的分支。
3. 做小而完整的提交；共享接口和数据库变更及时同步。
4. 提交 PR，写清行为变化、验证命令及实际结果。未执行的检查写明原因。
5. CI 通过、另一人完成评审、评审意见处理完毕后，两人中任意一人都可以合并。PR 作者也可以在同伴批准后执行合并。
6. 默认使用 Squash merge，合并后删除已完成的远程功能分支，并同步本地 `main`。

```bash
cd ~/projects/slm-from-scratch
git switch main
git pull --ff-only origin main
git switch -c feature/training-status

# 修改并完成对应检查后
# 按本次任务替换为实际修改的文件
git add -- scripts/train.py tests/test_train_cli.py
git commit -m "feat: describe the change"
git push -u origin HEAD
```

人类开发者可用 `feature/`、`fix/`、`docs/` 分支；Codex 创建的分支使用 `codex/`。

交叉评审是团队协作约定。核对当日 `main` 没有分支保护或 ruleset，因此该约定尚未由 GitHub 强制执行。若之后启用保护，要求通过 PR、一次同伴批准及所需 CI，并保留两人的合并权限；不设单人专属合并人或 CODEOWNERS 审批限制。

## 降低双人并行冲突

- 同一时间尽量不让两人改同一个大文件，尤其是 `platform/backend/src/main.cpp` 和 `platform/frontend/src/App.tsx`。
- API、数据库迁移、`pyproject.toml`、前端依赖和 Compose 属于共享接口，先在任务中写明影响。
- 数据库变更新增迁移，不改写已被另一人使用的迁移；先约定编号与执行顺序。
- 接口变更给出 JSON 样例、状态码、空值和错误约定；前后端各自准备验证用例。
- 自己分支需要更新时先 `git fetch origin`，再 `git merge origin/main`。两人一起处理有业务含义的冲突，不覆盖同伴提交。
- 本地 `.env`、数据集、模型权重和训练日志不提交。数据与 Checkpoint 沿用仓库外存放策略。

## 按修改范围验证

### Python 模型、训练或采集器

使用 Python 3.12，与 Python CI 保持一致。先安装适合本机的 PyTorch，再执行：

```bash
python -m pip install -e ".[dev,monitor]"
python -m compileall -q src scripts tests
python -m pytest tests/ -v
git diff --check
```

CUDA 冒烟测试只在具备 GPU 的机器上补充执行；CPU 自动化测试是基础检查。报告本次实际结果，不沿用文档中的历史通过数量。

### 后端、数据库或 Compose

先参考 `.env.example` 在本地设置数据库环境变量。执行：

```bash
docker compose config --quiet
docker compose build backend
docker compose up -d --wait postgres
```

新建本地数据库需要按顺序应用 `platform/database/migrations/` 中的迁移，命令见 [平台说明](docs/platform-v1.md)。已有数据库先确认迁移记录，避免重复执行。然后启动后端并检查 API：

```bash
docker compose up -d backend
python scripts/check_backend_health.py
python scripts/check_training_runs_api.py
python scripts/check_training_metrics_api.py
python scripts/check_hardware_snapshots_api.py
```

训练列表和指标检查脚本需要预置测试记录，具体方式见 `.github/workflows/backend-ci.yml`。应在独立开发数据库或 CI 上执行会写入数据的检查。

### React 前端

```bash
cd platform/frontend
npm ci
npm run lint
npm run build
```

界面改动还需连接后端，检查有数据、空数据、请求失败，以及切换实验后不显示上一实验的数据。当前 GitHub Actions 包含 Python 和后端检查，前端 lint/build 需在 PR 中提供本地结果。

### 仅修改文档或模板

核对命令、目录、相对链接和任务状态，并执行 `git diff --check`；不要求重跑训练或构建镜像。

## 完成标准

- 对应需求的验收条件全部满足。
- 相关自动化检查通过，联调结果和未执行项可追溯。
- 文档区分已实现能力、已验证结果和下一步计划。
- 同伴评审完成，两人均可执行最后的 PR 合并。
