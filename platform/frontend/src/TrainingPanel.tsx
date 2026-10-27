
import { useEffect, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Database,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./TrainingPanel.css";

type TrainingRun = {
  id: number;
  name: string;
  model_name: string;
  dataset_name: string;
  status: string;
};

type TrainingMetric = {
  global_step: number;
  epoch: number | null;
  train_loss: number | null;
  val_loss: number | null;
  learning_rate: number | null;
  tokens_per_second: number | null;
};

type ApiList<T> = {
  status: string;
  count: number;
  items: T[];
};

function showNumber(
  value: number | null | undefined,
  digits = 2,
): string {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function showLearningRate(
  value: number | null | undefined,
): string {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value.toExponential(2)
    : "—";
}

export default function TrainingPanel() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] =
    useState<number | null>(null);

  const [metricsResult, setMetricsResult] = useState<{
    runId: number;
    items: TrainingMetric[];
  } | null>(null);

  const [runsError, setRunsError] = useState("");
  const [metricsError, setMetricsError] = useState("");

  // 每 15 秒刷新一次训练实验列表
  useEffect(() => {
    const controller = new AbortController();

    async function loadRuns() {
      try {
        const response = await fetch(
          "/api/training-runs",
          {
            signal: controller.signal,
            cache: "no-store",
          },
        );

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data =
          (await response.json()) as ApiList<TrainingRun>;

        if (data.status !== "ok" ||
            !Array.isArray(data.items)) {
          throw new Error("训练实验列表格式异常");
        }

        if (controller.signal.aborted) return;

        setRuns(data.items);
        setRunsError("");

        // 保留用户当前选择；若不存在则选择第一项
        setSelectedRunId((current) => {
          if (
            current !== null &&
            data.items.some((run) => run.id === current)
          ) {
            return current;
          }

          return data.items[0]?.id ?? null;
        });
      } catch (error) {
        if (controller.signal.aborted) return;

        setRunsError(
          error instanceof Error
            ? error.message
            : "无法获取训练实验",
        );
      }
    }

    const initialTimer = window.setTimeout(() => {
      void loadRuns();
    }, 0);

    const refreshTimer = window.setInterval(() => {
      void loadRuns();
    }, 15000);

    return () => {
      controller.abort();
      window.clearTimeout(initialTimer);
      window.clearInterval(refreshTimer);
    };
  }, []);

  // 根据选中的实验 ID 加载指标
  useEffect(() => {
    if (selectedRunId === null) return;

    const runId = selectedRunId;
    const controller = new AbortController();

    async function loadMetrics() {
      try {
        const response = await fetch(
          `/api/training-runs/${runId}/metrics`,
          {
            signal: controller.signal,
            cache: "no-store",
          },
        );

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data =
          (await response.json()) as ApiList<TrainingMetric>;

        if (data.status !== "ok" ||
            !Array.isArray(data.items)) {
          throw new Error("训练指标数据格式异常");
        }

        if (controller.signal.aborted) return;

        const sorted = [...data.items].sort(
          (a, b) => a.global_step - b.global_step,
        );

        setMetricsResult({
          runId,
          items: sorted,
        });
        setMetricsError("");
      } catch (error) {
        if (controller.signal.aborted) return;

        setMetricsError(
          error instanceof Error
            ? error.message
            : "无法获取训练指标",
        );
      }
    }

    const initialTimer = window.setTimeout(() => {
      void loadMetrics();
    }, 0);

    const refreshTimer = window.setInterval(() => {
      void loadMetrics();
    }, 5000);

    return () => {
      controller.abort();
      window.clearTimeout(initialTimer);
      window.clearInterval(refreshTimer);
    };
  }, [selectedRunId]);

  const selectedRun = runs.find(
    (run) => run.id === selectedRunId,
  );

  // 确保不显示上一实验遗留的指标
  const metrics =
    metricsResult?.runId === selectedRunId
      ? metricsResult.items
      : [];

  const latest = metrics.at(-1);

  return (
    <section
      className="section-block"
      id="training-monitor"
    >
      <div className="section-heading">
        <div>
          <div className="eyebrow">
            04 / MODEL TRAINING
          </div>

          <h2>模型训练监控</h2>

          <p>
            关联训练实验、Loss、学习率与
            Tokens/s，分析训练过程。
          </p>
        </div>

        <div className="training-title-icon">
          <BrainCircuit size={22} />
        </div>
      </div>

      <div className="training-panel">
        <div className="training-toolbar">
          <div className="training-toolbar-label">
            <Database size={16} />
            <span>TRAINING RUN</span>
          </div>

          <select
            aria-label="选择训练实验"
            value={selectedRunId ?? ""}
            disabled={runs.length === 0}
            onChange={(event) => {
              setSelectedRunId(
                Number(event.target.value),
              );
            }}
          >
            {runs.length === 0 && (
              <option value="">
                暂无训练实验
              </option>
            )}

            {runs.map((run) => (
              <option
                key={run.id}
                value={run.id}
              >
                #{run.id} · {run.name}
              </option>
            ))}
          </select>
        </div>

        {runsError && (
          <div className="training-error">
            实验列表获取失败：{runsError}
          </div>
        )}

        {runs.length === 0 && !runsError ? (
          <div className="training-empty">
            <BrainCircuit size={36} />

            <h3>等待首次模型训练</h3>

            <p>
              当前数据库尚无训练实验记录。
              以后创建训练实验并上传指标后，
              此处会自动展示真实训练数据。
            </p>

            <span>
              API: GET /api/training-runs
            </span>
          </div>
        ) : selectedRun ? (
          <>
            <div className="training-run-info">
              <span>
                实验：{selectedRun.name}
              </span>

              <span>
                模型：{selectedRun.model_name}
              </span>

              <span>
                状态：{selectedRun.status}
              </span>
            </div>

            {metricsError && (
              <div className="training-error">
                指标获取失败：{metricsError}
              </div>
            )}

            <div className="training-stats">
              <div>
                <span>GLOBAL STEP</span>
                <strong>
                  {latest?.global_step ?? "—"}
                </strong>
              </div>

              <div>
                <span>TRAIN LOSS</span>
                <strong>
                  {showNumber(latest?.train_loss, 4)}
                </strong>
              </div>

              <div>
                <span>VAL LOSS</span>
                <strong>
                  {showNumber(latest?.val_loss, 4)}
                </strong>
              </div>

              <div>
                <span>LEARNING RATE</span>
                <strong>
                  {showLearningRate(
                    latest?.learning_rate,
                  )}
                </strong>
              </div>

              <div>
                <span>TOKENS / SEC</span>
                <strong>
                  {showNumber(
                    latest?.tokens_per_second,
                    1,
                  )}
                </strong>
              </div>
            </div>

            {metrics.length === 0 ? (
              <div className="training-empty">
                <Activity size={30} />
                <h3>等待训练指标</h3>
                <p>
                  该实验尚无可展示的训练指标。
                </p>
              </div>
            ) : (
              <div className="training-charts">
                <div className="training-chart-card">
                  <h3>Loss 变化趋势</h3>

                  <div className="training-chart">
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >
                      <LineChart data={metrics}>
                        <CartesianGrid
                          stroke="#1A3048"
                          strokeDasharray="4 4"
                        />

                        <XAxis
                          dataKey="global_step"
                          type="number"
                          domain={[
                            "dataMin",
                            "dataMax",
                          ]}
                          tick={{
                            fill: "#8199B5",
                            fontSize: 10,
                          }}
                        />

                        <YAxis
                          tick={{
                            fill: "#8199B5",
                            fontSize: 10,
                          }}
                        />

                        <Tooltip />

                        <Line
                          type="monotone"
                          dataKey="train_loss"
                          name="Train Loss"
                          stroke="#54E7FF"
                          strokeWidth={2}
                          dot={false}
                          connectNulls={false}
                        />

                        <Line
                          type="monotone"
                          dataKey="val_loss"
                          name="Val Loss"
                          stroke="#FFBD6B"
                          strokeWidth={2}
                          dot={false}
                          connectNulls={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="training-chart-card">
                  <h3>训练吞吐趋势</h3>

                  <div className="training-chart">
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >
                      <LineChart data={metrics}>
                        <CartesianGrid
                          stroke="#1A3048"
                          strokeDasharray="4 4"
                        />

                        <XAxis
                          dataKey="global_step"
                          type="number"
                          domain={[
                            "dataMin",
                            "dataMax",
                          ]}
                          tick={{
                            fill: "#8199B5",
                            fontSize: 10,
                          }}
                        />

                        <YAxis
                          tick={{
                            fill: "#8199B5",
                            fontSize: 10,
                          }}
                        />

                        <Tooltip />

                        <Line
                          type="monotone"
                          dataKey="tokens_per_second"
                          name="Tokens/s"
                          stroke="#48E3B5"
                          strokeWidth={2}
                          dot={false}
                          connectNulls={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : null}

        <div className="training-footnote">
          训练指标每 5 秒自动查询一次。
          不生成模拟训练数据。
        </div>
      </div>
    </section>
  );
}
