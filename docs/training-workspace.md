# 训练工作台与模型内部观测

前端默认进入训练工作台，侧栏可切换到原有硬件监控。工作台使用真实实验和指标，提供历史回看、学习曲线、内部诊断及 CSV 导出。

## 首次使用与升级

首次安装按 [平台说明](platform-v1.md) 配置数据库并依次应用全部迁移。已应用 `001` 和 `002` 的环境需新增 `003` 和 `004`；已应用 `003` 的环境只需新增 `004`。升级时先停止旧后端和训练上报，迁移后重新构建服务。以下升级命令假设已经执行 `003`，尚未执行 `004`：

```bash
docker compose stop backend
docker compose exec -T postgres sh -c \
  'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < platform/database/migrations/004_training_monitor.sql
docker compose up -d --build backend frontend
```

`003` 新增可为空的 `diagnostics JSONB`，已有实验和指标保留。旧脚本仍可上报原来的指标，新后端需要先完成迁移。当前没有自动迁移记录表，已执行的 SQL 不要重复执行。

## 开始一段可观察的训练

在已安装 PyTorch 和项目依赖的 Python 环境执行：

```bash
python scripts/train.py \
  --epochs 3 --batch-size 8 --max-steps 30 \
  --learning-rate 0.0005 \
  --api-base-url http://127.0.0.1:8080 \
  --run-name tinygpt-observed \
  --diagnostics-interval 10
```

不提供 `--data-path` 时使用脚本内置的小型演示文本；实际训练请传入本地 UTF-8 文本路径。浏览器打开 `http://127.0.0.1:5173` 并选择此次实验。

实验选择会同步到地址栏的 `?run=<id>`，可复制链接或刷新后继续查看该实验。单独运行前端开发服务时，可用 `SLM_API_PROXY_TARGET=http://127.0.0.1:8080 npm run dev` 指向本机后端；Compose 默认仍使用 `backend:8080`。

`--diagnostics-interval` 默认 `10`；显式传入 `0` 可关闭内部采样。大于零时，在第一步、每 N 步、每个 Epoch 最后一步和 `--max-steps` 结束步骤采样。浏览器只展示上报结果，训练在 Python 中执行。

采样会保存**首个样本的短输入文本、目标字符和预测 Token**，请使用适合在本地界面展示的数据。关闭时不增加诊断 hooks 或参数副本；开启时短暂复制参数以测量真实更新量，并读取 GPU 数值，会增加内存和同步开销，不适合在大型模型上高频采样。

## 训练概览与历史回看

- 显示模型层数、注意力头数、参数量、批大小、上下文长度、设备及原始配置；旧实验缺少配置时显示 `—`。
- 进度为已上报的最新步数 / 计划步数；计划步数取 Epoch 总批次数与 `--max-steps` 的较小值。
- 显示后端记录的状态和创建/开始/结束时间。训练异常自动上报 `failed`，键盘中断上报 `cancelled`；若 API 本身不可达则输出上报失败信息。
- 滑块按实际步骤回看，或播放已有训练记录；点击“回到最新”恢复跟随。切换实验退出回看并重置分页。
- 基础指标每 5 秒查询，实验列表每 15 秒查询；可暂停自动更新、手动刷新。请求失败时说明是否保留了旧数据。

| 图表或统计 | 含义与计算 |
| --- | --- |
| Train Loss | 对应 Batch 在更新前计算的交叉熵 |
| Val Loss | 最近一次实际评估的验证损失，明确标注评估 Step |
| Loss change | 截至回看位置，相对已加载窗口首个有效训练 Loss 的下降百分比；负数为上升，基准为零时不计算 |
| EMA | `EMA_t = α × loss_t + (1-α) × EMA_(t-1)`，可关闭或调整平滑强度 |
| Perplexity | `exp(Train Loss)`，溢出时显示缺失值；不是准确率，仅对相同词表和数据可比 |
| Tokens/s | 真实目标 Token 数 / 训练步耗时，包含诊断开销，不包含 HTTP 上报和整轮验证 |
| 验证差值 | 最近验证值减去同一步训练 Batch Loss；评估数据与时机不同，单点不能确定是否过拟合 |
| Epoch 汇总 | 已加载窗口内截至回看位置的 Step 范围、末步 Loss、最近验证和平均吞吐 |

验证只在实际评估位置绘点，不连接缺失值。图表保留完整已加载曲线作对照，竖线为回看位置；面板统计按回看位置更新。逐步记录每页 20 条，点击 Step 可定位回看。CSV 导出截至回看位置的基础指标，保留空值，不包含文本片段和注意力矩阵。

接口最多返回最近 **1000 条**指标、最近 **100 个**实验。首次图表加载最近 1000 条，页面可通过“加载更早 1000 步”继续读取历史。曲线、回看和 CSV 基于已加载范围；独立监控卡片使用后端全程统计。

## 模型内部变化

五个可点击阶段解释数据流：分词 → 词嵌入与位置 → Transformer → 预测与损失 → 反向传播和 AdamW。阶段同时显示真实采样的输入 ID、张量形状及耗时；未采集的旧实验明确显示缺失。输入阶段耗时为当前批次设备传输，分词发生在构建数据集时；反向传播和优化器分别计时。

内部诊断取自同一次真实训练 forward，不增加额外推理：

| 采样项目 | 取值范围与时机 |
| --- | --- |
| 逐层激活均值/标准差 | Embeddings、每个 Transformer block、最终 LayerNorm、语言模型头；本次前向的整个 Batch，标准差使用总体标准差 |
| 梯度 L2 范数 | 反向传播后、更新前；按上述模块聚合全部可训练参数 |
| 参数 L2 范数 | AdamW 更新后 |
| 实际更新 L2 范数 | 更新前后参数差值的范数，包括优化器和权重衰减的实际影响 |
| Token 准确率 | 更新前，整个 Batch 全部位置的 argmax 预测与目标比较 |
| 预测熵 | 首个样本最后位置，单位 nats |
| Top 5 概率 | 首个样本最后位置预测的下一字符，同时显示输入片段与真实目标；不同步骤的样本可能不同，Top 5 不必合计 100% |
| 注意力热力图 | 第一层、第一头、首个样本最多 16 个位置，dropout 前权重；行为查询 Token、列为被关注 Token，未来位置有因果掩码，点击查看数值 |

回看位置未采样时，只显示此前最近一次诊断并标明真实采样 Step，不使用未来快照。旧实验尚无诊断时显示开启方法。单次快照或某层范数大小不能独立证明模型训练稳定或已经学会某种能力。

## 接口扩展与验证

`POST /api/training-runs` 接受可选 `config` 对象（最多 16 KiB），写入已有配置列；实验列表返回它。训练脚本写入结构、超参数、计划步数和采样间隔。

`POST /api/training-runs/{run_id}/metrics` 接受可选 `diagnostics` 对象（最多 64 KiB）或 `null`；查询接口也返回该字段，旧数据为 `null`。诊断对象使用 `version: 1`，包含 `layers`、全模型范数、Token 准确率、熵、`prediction` 和 `attention`。前端校验版本、数组尺寸、有效数值和概率范围。

```bash
python -m pytest tests/ -v
python scripts/check_training_diagnostics_api.py
cd platform/frontend
npm ci
npm run lint
npm test
npm run build
```

API 检查脚本会创建测试实验和少量指标，应在独立开发数据库或 CI 中执行。后端 CI 加入 `003` 迁移和诊断 API 检查；前端 CI 运行 lint、数据计算测试和生产构建。

浏览器验收覆盖真实训练诊断、旧实验、无实验、无指标、API 失败与恢复、快速切换、回看播放、CSV 导出及窄屏布局。

### 本次本地验收记录（2026-10-28）

- Python 3.14 本机环境：22 项自动化测试通过，包括诊断与原始训练更新结果一致、hook 异常清理、CLI 配置与采样间隔上报。
- 前端：lint 为 0 错误、0 警告，3 项数据计算测试通过，TypeScript 与 Vite 生产构建通过；现有 Recharts 主包仍有超过 500 kB 的体积提示。
- C++：直接编译、链接并运行真实后端；对独立 PostgreSQL 18 数据库依次应用三份迁移，原有健康/实验/指标/硬件 API 检查及新增诊断往返检查均通过。Compose 使用 PostgreSQL 16，容器构建由后端 CI 再验证。
- 真实 CPU 冒烟训练：内置小文本、2 个 Epoch 上限、最多 60 步、每 5 步采样，实际得到 60 条训练指标和 14 次诊断；没有保存 Checkpoint。
- 浏览器：已核对真实训练页面、16×16 注意力、逐层诊断、历史播放、分页、61 行 CSV（含表头），以及无实验、无指标、旧实验、快速切换、错误保留数据与恢复、390 px 窄屏布局。

以上为小规模功能验收，不代表大型模型性能或通用语言能力评测。

## 完整监控接入（2026-10-30）

新增 `004_training_monitor.sql`，以附加列存储训练进程的最近上报阶段与时间，并对内部诊断建立部分索引。保留已有数据，不重写历史迁移。

| 接口 | 行为 |
| --- | --- |
| `POST /api/training-runs/{id}/runtime` | 进程上报 `training` / `validation` / `checkpoint` / `completed` / `failed`，含非负 `global_step` 和 `epoch`，最多 8 KiB；仅 `running` 实验可写入，其他状态返回 409 |
| `GET /api/training-runs/{id}/monitor` | 实验配置、最近运行阶段和数据库时间、全程记录数、最新步骤、最低损失、首末损失变化、平均吞吐、诊断次数与最新诊断 |
| `GET /api/training-runs/{id}/diagnostics` | 独立查询诊断；`at_step` 为包含该步骤的回看上界，`before_step` 为排他的分页游标，`limit` 为 1–1000 |
| `GET /api/training-runs/{id}/metrics/history` | 指标历史分页，`before_step` 为排他的游标，`limit` 为 1–1000；返回升序 `items`、`has_more`、`next_before_step` |

查询不存在的实验返回 404，无指标实验返回空列表和空统计值。无效参数返回 400。原来的最近 1000 条指标接口保持兼容。

诊断仍使用 `version: 1`，新增 `pipeline`（毫秒）、`input_shape`、`shapes`、`input_ids` 和 `target_ids`。耗时来自同一次训练更新；CUDA 采样会同步设备以测量实际执行耗时。吞吐不包含 HTTP 上报与整轮验证。

旧实验未采集过的输入、梯度、注意力及运行阶段不能事后还原；不会为旧记录填造数据。新训练自动上报完整结构、样本数、滑窗步长、优化器、权重衰减和默认诊断采样。

新增 API 验收：`SLM_API_BASE_URL=<isolated backend> python scripts/check_training_monitor_api.py`。该脚本会创建 1002 条测试指标，须在隔离开发数据库或 CI 上运行。

### 本次完整链路验收

- Python 3.12 / CPU：35 项测试通过，覆盖原始训练数值不变、诊断 hook 清理、CLI 阶段上报及异常收尾。
- 前端：lint 为 0 错误、0 警告，5 项数据测试、TypeScript 和 Vite 构建通过。
- C++：Docker 镜像构建成功；临时 PostgreSQL 16 上应用全部迁移，原诊断 API 与新监控 API 验收通过，包括 1002 条指标、游标分页、窗口外诊断、空实验、无效参数及终态拒写。
- 真实训练：CPU、内置小文本、最多 4 步，得到 4 条指标和 2 次完整内部采样；未保存 Checkpoint、未启动正式训练。
- 本地服务：已应用 `004` 并更新后端，8080 直接访问和 5173 前端代理均返回真实监控数据。可见验证实验为 `monitor-connection-check-20261030`（实验 #19）。
- 浏览器：真实阶段耗时、输入和张量形状、逐层数据、注意力展示、旧实验隔离、加载更早历史、空指标、错误保留与恢复通过；390 px 页面没有水平溢出。

硬件数据由已有 `scripts/hardware_monitor.py` 采集并每 5 秒上报，浏览器每 5 秒读取。硬件进程停止后，页面会按既有规则显示采样过期。启动命令：`.venv/bin/python scripts/hardware_monitor.py --base-url http://127.0.0.1:8080 --interval 5`。持续采集时不绑定已经结束的验证实验。
