import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Download,
  FlaskConical,
  Layers,
  Pause,
  Play,
  Radio,
  RefreshCw,
  SlidersHorizontal,
  Terminal,
} from "lucide-react";
import {
  buildSeries,
  epochSummaries,
  metricsCsv,
  numeric,
  summarize,
} from "./training-data";
import { number, scientific, dateText } from "./training-format";
import useTrainingData from "./useTrainingData";
import Internals from "./TrainingInternals";
import { Stat, Plot } from "./TrainingCharts";
import "./TrainingPanel.css";

const statusLabels: Record<string, string> = {
  created: "等待开始",
  running: "训练中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
export default function TrainingPanel() {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [revision, setRevision] = useState(0);
  const { runs, selectedId, setSelectedId, runsState, result } =
    useTrainingData(autoRefresh, revision);
  const [tab, setTab] = useState("overview");
  const [step, setStep] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [alpha, setAlpha] = useState(0.2);
  const [page, setPage] = useState(0);
  const run = runs.find((item) => item.id === selectedId);
  const metrics = useMemo(() => result?.items ?? [], [result?.items]);
  const current =
    (step === null
      ? metrics.at(-1)
      : metrics.find((m) => m.global_step === step)) ?? metrics.at(-1);
  const currentIndex = current ? metrics.indexOf(current) : -1;
  const visible = metrics.slice(0, currentIndex + 1);
  const series = useMemo(() => buildSeries(metrics, alpha), [metrics, alpha]);
  const summary = summarize(visible);
  const epochs = epochSummaries(visible);
  const planned = numeric(run?.config.planned_steps);
  const latestStep = metrics.at(-1)?.global_step;
  const progress =
    planned !== null && planned > 0 && latestStep !== undefined
      ? Math.min(100, (latestStep / planned) * 100)
      : null;
  const diagnosticCount = metrics.filter((m) => m.diagnostics).length;
  const lastValidation = summary.validation;
  const gap =
    lastValidation && lastValidation.train_loss !== null
      ? lastValidation.val_loss! - lastValidation.train_loss
      : null;
  const pages = Math.max(1, Math.ceil(metrics.length / 20));
  const actualPage = Math.min(page, pages - 1);
  const rows = [...metrics]
    .reverse()
    .slice(actualPage * 20, actualPage * 20 + 20);

  useEffect(() => {
    if (!playing || !metrics.length) return;
    const timer = window.setTimeout(() => {
      if (currentIndex >= metrics.length - 1) setPlaying(false);
      else setStep(metrics[currentIndex + 1].global_step);
    }, 850);
    return () => window.clearTimeout(timer);
  }, [playing, currentIndex, metrics]);

  function chooseRun(id: number) {
    const url = new URL(window.location.href);
    url.searchParams.set("run", String(id));
    window.history.replaceState(null, "", url);
    setSelectedId(id);
    setStep(null);
    setPlaying(false);
    setPage(0);
  }
  function exportCsv() {
    const url = URL.createObjectURL(
      new Blob(["\ufeff" + metricsCsv(visible)], {
        type: "text/csv;charset=utf-8",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `training-${selectedId}-through-step-${current?.global_step ?? 0}.csv`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <section className="training-lab" id="training-monitor">
      <header className="lab-hero">
        <div>
          <div className="lab-kicker">
            <span /> TRAINING WORKSPACE
          </div>
          <h1>
            看见模型，<span>一步步学会。</span>
          </h1>
          <p>从第一步预测到每一次参数更新，观察 TinyGPT 的训练轨迹。</p>
        </div>
        <div className="lab-hero-mark">
          <BrainCircuit size={52} strokeWidth={1} />
          <span>TinyGPT / FROM SCRATCH</span>
        </div>
      </header>
      <div className="lab-toolbar">
        <div className="lab-experiment-picker">
          <FlaskConical size={18} />
          <label htmlFor="training-run">训练实验</label>
          <select
            id="training-run"
            value={selectedId ?? ""}
            onChange={(event) => chooseRun(Number(event.target.value))}
            disabled={!runs.length}
          >
            <option value="" disabled>
              {runsState.loading ? "正在连接…" : "选择实验"}
            </option>
            {runs.map((item) => (
              <option value={item.id} key={item.id}>
                #{item.id} · {item.name}
              </option>
            ))}
          </select>
        </div>
        <div className="lab-toolbar-actions">
          <button
            type="button"
            className={`lab-button ${autoRefresh ? "toggled" : ""}`}
            aria-pressed={autoRefresh}
            onClick={() => setAutoRefresh((v) => !v)}
          >
            <Radio size={14} />
            {autoRefresh ? "自动更新" : "已暂停更新"}
          </button>
          <button
            className="lab-button icon-only"
            type="button"
            onClick={() => setRevision((v) => v + 1)}
            aria-label="刷新训练数据"
          >
            <RefreshCw size={15} className={result?.loading ? "spin" : ""} />
          </button>
          <button
            className="lab-button"
            type="button"
            onClick={exportCsv}
            disabled={!visible.length}
          >
            <Download size={14} />
            导出 CSV
          </button>
        </div>
      </div>
      {runsState.error && (
        <div className="lab-error" role="alert">
          实验列表获取失败：{runsState.error}。
          {runs.length > 0
            ? "以下保留上次成功获取的列表。"
            : "请检查服务连接后重试。"}
          <button type="button" onClick={() => setRevision((v) => v + 1)}>
            重试
          </button>
        </div>
      )}
      {!run ? (
        <div className="lab-empty lab-card">
          <FlaskConical size={40} />
          <h2>
            {runsState.loading
              ? "正在加载训练实验"
              : runsState.error
                ? "训练服务暂不可用"
                : "从第一次训练开始"}
          </h2>
          <p>
            {runsState.loading
              ? "正在读取实验列表，请稍候。"
              : "启动一次带指标上报的训练，实验和曲线会自动出现在这里。"}
          </p>
          {!runsState.loading && (
            <code>
              python scripts/train.py --max-steps 30 --api-base-url
              http://127.0.0.1:8080 --diagnostics-interval 10
            </code>
          )}
        </div>
      ) : (
        <>
          <div className="lab-run-heading">
            <div>
              <div className="lab-run-title">
                <h2>{run.name}</h2>
                <span className={`lab-status status-${run.status}`}>
                  <i />
                  {statusLabels[run.status] ?? run.status}
                </span>
              </div>
              <p>
                <span>{run.model_name}</span>
                <span title={run.dataset_name}>{run.dataset_name}</span>
                <span>实验 #{run.id}</span>
              </p>
            </div>
            <div className="lab-updated">
              最近同步
              <strong>
                {result?.updated
                  ? result.updated.toLocaleTimeString("zh-CN", {
                      hour12: false,
                    })
                  : "等待指标"}
              </strong>
            </div>
          </div>
          {result?.error && (
            <div className="lab-error" role="alert">
              指标获取失败：{result.error}。
              {metrics.length
                ? "当前展示上次成功获取的数据，可能已过期。"
                : "暂时没有可展示的指标。"}
              <button type="button" onClick={() => setRevision((v) => v + 1)}>
                重试
              </button>
            </div>
          )}
          <div className="lab-run-meta">
            <div>
              <span>模型结构</span>
              <strong>
                {number(run.config.n_layers, 0)} 层 /{" "}
                {number(run.config.n_heads, 0)} 注意力头
              </strong>
            </div>
            <div>
              <span>参数量</span>
              <strong>{number(run.config.parameter_count, 0)}</strong>
            </div>
            <div>
              <span>批大小 / 上下文</span>
              <strong>
                {number(run.config.batch_size, 0)} /{" "}
                {number(run.config.context_length, 0)}
              </strong>
            </div>
            <div>
              <span>训练设备</span>
              <strong>
                {typeof run.config.device === "string"
                  ? run.config.device
                  : "—"}
              </strong>
            </div>
            <div className="lab-progress">
              <span>
                已上报进度{" "}
                <b>
                  {progress === null
                    ? "总步数未记录"
                    : `${number(progress, 1)}%`}
                </b>
              </span>
              <div>
                <i style={{ width: `${progress ?? 0}%` }} />
              </div>
              <small>
                {number(latestStep, 0)} / {number(planned, 0)} steps
              </small>
            </div>
          </div>
          {!metrics.length ? (
            <div className="lab-empty lab-card">
              <Activity size={32} />
              <h3>
                {result?.loading || !result
                  ? "正在读取训练指标"
                  : result.error
                    ? "暂时无法读取指标"
                    : "实验已创建，等待第一个 Step"}
              </h3>
              <p>训练脚本上报指标后，可查看学习曲线和内部变化。</p>
            </div>
          ) : (
            current && (
              <>
                <section className="lab-replay">
                  <div className="lab-replay-title">
                    <span className={step === null ? "lab-live" : ""}>
                      {step === null ? <Radio size={15} /> : <Play size={14} />}
                      {step === null ? "跟随最新" : "历史回看"}
                    </span>
                    <strong>STEP {number(current.global_step, 0)}</strong>
                    <small>EPOCH {number(current.epoch, 0)}</small>
                  </div>
                  <div className="lab-replay-controls">
                    <button
                      className="lab-button icon-only"
                      aria-label={playing ? "暂停回放" : "播放训练历史"}
                      type="button"
                      disabled={metrics.length < 2}
                      onClick={() => {
                        if (!playing && currentIndex >= metrics.length - 1)
                          setStep(metrics[0].global_step);
                        setPlaying((v) => !v);
                      }}
                    >
                      {playing ? <Pause size={15} /> : <Play size={15} />}
                    </button>
                    <input
                      aria-label="回看训练步骤"
                      type="range"
                      min={0}
                      max={metrics.length - 1}
                      value={currentIndex}
                      onChange={(event) => {
                        setStep(
                          metrics[Number(event.target.value)].global_step,
                        );
                        setPlaying(false);
                      }}
                    />
                    <button
                      type="button"
                      className="lab-button"
                      disabled={step === null}
                      onClick={() => {
                        setStep(null);
                        setPlaying(false);
                      }}
                    >
                      回到最新
                    </button>
                  </div>
                  <p>
                    拖动查看已记录步骤；播放为历史回放。已加载 {metrics.length}{" "}
                    条，范围 Step {metrics[0].global_step}–{latestStep}
                    （接口最多返回最近 1000 条）。
                  </p>
                </section>
                <div className="lab-stats">
                  <Stat
                    label="训练损失 / TRAIN LOSS"
                    value={number(current.train_loss, 4)}
                    detail={`本批次 · Step ${current.global_step}`}
                    accent
                  />
                  <Stat
                    label="验证损失 / VAL LOSS"
                    value={number(lastValidation?.val_loss, 4)}
                    detail={
                      lastValidation
                        ? `最近验证 · Step ${lastValidation.global_step}`
                        : "尚无验证记录"
                    }
                  />
                  <Stat
                    label="损失变化 / LOSS CHANGE"
                    value={
                      summary.improvement === null
                        ? "—"
                        : `${number(summary.improvement, 1)}%`
                    }
                    detail={
                      <>
                        <ArrowDownRight size={12} /> 相对已加载窗口首个有效
                        Loss；负值为上升
                      </>
                    }
                  />
                  <Stat
                    label="训练吞吐 / TOKENS/S"
                    value={number(current.tokens_per_second, 1)}
                    detail={`窗口均值 ${number(summary.averageSpeed, 1)} tokens/s`}
                  />
                </div>
                <nav className="lab-tabs" aria-label="训练视图">
                  {[
                    { id: "overview", label: "训练概览", icon: Activity },
                    { id: "internals", label: "模型内部变化", icon: Layers },
                    { id: "records", label: "逐步记录", icon: Terminal },
                  ].map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={tab === item.id ? "active" : ""}
                      aria-pressed={tab === item.id}
                      onClick={() => setTab(item.id)}
                    >
                      <item.icon size={16} />
                      {item.label}
                      {item.id === "internals" && (
                        <small>{diagnosticCount} 次采样</small>
                      )}
                    </button>
                  ))}
                </nav>
                {tab === "overview" && (
                  <div className="lab-overview">
                    <div className="lab-analysis-grid">
                      <div>
                        <div className="lab-chart-tools">
                          <span>
                            <SlidersHorizontal size={14} /> Loss 平滑
                          </span>
                          <select
                            aria-label="Loss 平滑系数"
                            value={alpha}
                            onChange={(event) =>
                              setAlpha(Number(event.target.value))
                            }
                          >
                            <option value={1}>关闭平滑</option>
                            <option value={0.4}>轻度 · α 0.4</option>
                            <option value={0.2}>中度 · α 0.2</option>
                            <option value={0.05}>强度 · α 0.05</option>
                          </select>
                        </div>
                        <Plot
                          title="模型如何逐步减少预测误差"
                          subtitle="原始训练 Loss、指数移动平均与独立验证记录"
                          data={series}
                          selected={current.global_step}
                          lines={[
                            {
                              key: "train_loss",
                              name: "Train Loss",
                              color: "#6b8792",
                              dashed: true,
                            },
                            ...(alpha === 1
                              ? []
                              : [
                                  {
                                    key: "smoothed_loss",
                                    name: "EMA",
                                    color: "#76d7be",
                                  },
                                ]),
                            {
                              key: "val_loss",
                              name: "Val Loss",
                              color: "#f0bd7b",
                            },
                          ]}
                        />
                        <p className="lab-note">
                          EMA
                          仅用于观察趋势；验证只在实际评估步骤绘点，空缺不连线。竖线是当前回看步骤，完整曲线保留作对照。
                        </p>
                      </div>
                      <aside className="lab-reading lab-card">
                        <span className="lab-kicker">READ THE TRAINING</span>
                        <h3>这一刻，如何理解？</h3>
                        <dl>
                          <div>
                            <dt>已加载窗口最低训练 Loss</dt>
                            <dd>{number(summary.best, 4)}</dd>
                          </div>
                          <div>
                            <dt>当前学习率</dt>
                            <dd>{scientific(current.learning_rate)}</dd>
                          </div>
                          <div>
                            <dt>训练困惑度 exp(Loss)</dt>
                            <dd>
                              {number(
                                buildSeries([current], 1)[0].perplexity,
                                2,
                              )}
                            </dd>
                          </div>
                          <div>
                            <dt>最近验证与同一步训练 Loss 差值</dt>
                            <dd>
                              {gap === null
                                ? "—"
                                : `${gap >= 0 ? "+" : ""}${number(gap, 4)}`}
                            </dd>
                          </div>
                        </dl>
                        <p>
                          Loss
                          越低表示对应数据的预测误差越小。验证与训练的样本、评估时机不同，单个差值不能确定是否过拟合。
                        </p>
                        <p>
                          困惑度是 Loss
                          的指数变换，并非准确率。训练吞吐包含诊断采样开销，不包含
                          HTTP 上报与整轮验证。
                        </p>
                      </aside>
                    </div>
                    <div className="lab-small-charts">
                      <Plot
                        title="训练吞吐"
                        subtitle="当前批次实际目标 Token / 训练步耗时"
                        data={series}
                        selected={current.global_step}
                        compact
                        lines={[
                          {
                            key: "tokens_per_second",
                            name: "Tokens/s",
                            color: "#76d7be",
                          },
                        ]}
                      />
                      <Plot
                        title="学习率轨迹"
                        subtitle="每一步优化器实际使用的学习率"
                        data={series}
                        selected={current.global_step}
                        compact
                        lines={[
                          {
                            key: "learning_rate",
                            name: "Learning rate",
                            color: "#a5b4f0",
                          },
                        ]}
                      />
                      <Plot
                        title="训练困惑度"
                        subtitle="exp(Train Loss)，仅对相同词表和数据可比"
                        data={series}
                        selected={current.global_step}
                        compact
                        lines={[
                          {
                            key: "perplexity",
                            name: "Perplexity",
                            color: "#f0bd7b",
                          },
                        ]}
                      />
                    </div>
                    <section className="lab-card">
                      <div className="lab-card-heading">
                        <div>
                          <h3>Epoch 进程</h3>
                          <p>截至当前回看步骤，基于已加载记录汇总。</p>
                        </div>
                        <Layers size={18} />
                      </div>
                      <div className="lab-epoch-list">
                        {epochs.map((epoch) => (
                          <div key={epoch.epoch}>
                            <span className="epoch-number">
                              {String(epoch.epoch).padStart(2, "0")}
                            </span>
                            <div>
                              <strong>Epoch {epoch.epoch}</strong>
                              <small>
                                Step {epoch.firstStep}–{epoch.lastStep} ·{" "}
                                {epoch.count} 条记录
                              </small>
                            </div>
                            <div>
                              <span>末步 Train Loss</span>
                              <strong>
                                {number(
                                  epoch.lastStep === current.global_step
                                    ? current.train_loss
                                    : visible.find(
                                        (m) => m.global_step === epoch.lastStep,
                                      )?.train_loss,
                                  4,
                                )}
                              </strong>
                            </div>
                            <div>
                              <span>最近 Val Loss</span>
                              <strong>
                                {number(epoch.validation?.val_loss, 4)}
                              </strong>
                            </div>
                            <div>
                              <span>平均 Tokens/s</span>
                              <strong>{number(epoch.averageSpeed, 1)}</strong>
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  </div>
                )}
                {tab === "internals" && (
                  <Internals
                    key={run.id}
                    run={run}
                    current={current}
                    metrics={metrics}
                  />
                )}
                {tab === "records" && (
                  <section className="lab-card">
                    <div className="lab-card-heading">
                      <div>
                        <h3>每一次更新都有记录</h3>
                        <p>
                          最新步骤在前；点击一行回看该步，CSV
                          导出截至回看位置的已加载记录。
                        </p>
                      </div>
                      <Terminal size={19} />
                    </div>
                    <div className="lab-table-scroll">
                      <table className="lab-records">
                        <thead>
                          <tr>
                            {[
                              "STEP",
                              "EPOCH",
                              "TRAIN LOSS",
                              "VAL LOSS",
                              "LEARNING RATE",
                              "TOKENS/S",
                              "诊断",
                            ].map((label) => (
                              <th key={label}>{label}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((metric) => (
                            <tr
                              key={metric.global_step}
                              className={
                                current.global_step === metric.global_step
                                  ? "selected"
                                  : ""
                              }
                            >
                              <td>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setStep(metric.global_step);
                                    setPlaying(false);
                                  }}
                                >
                                  #{metric.global_step}
                                </button>
                              </td>
                              <td>{number(metric.epoch, 0)}</td>
                              <td>{number(metric.train_loss, 5)}</td>
                              <td>{number(metric.val_loss, 5)}</td>
                              <td>{scientific(metric.learning_rate)}</td>
                              <td>{number(metric.tokens_per_second, 2)}</td>
                              <td>
                                {metric.diagnostics ? (
                                  <span className="lab-sampled">已采样</span>
                                ) : (
                                  "—"
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="lab-pagination">
                      <span>
                        共 {metrics.length} 条 · 第 {actualPage + 1} / {pages}{" "}
                        页
                      </span>
                      <button
                        type="button"
                        className="lab-button icon-only"
                        aria-label="上一页"
                        disabled={actualPage === 0}
                        onClick={() => setPage(actualPage - 1)}
                      >
                        <ChevronLeft size={16} />
                      </button>
                      <button
                        type="button"
                        className="lab-button icon-only"
                        aria-label="下一页"
                        disabled={actualPage >= pages - 1}
                        onClick={() => setPage(actualPage + 1)}
                      >
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </section>
                )}
              </>
            )
          )}
          <details className="lab-config">
            <summary>实验配置与时间记录</summary>
            <div>
              <dl>
                <div>
                  <dt>创建时间</dt>
                  <dd>{dateText(run.created_at)}</dd>
                </div>
                <div>
                  <dt>开始时间</dt>
                  <dd>{dateText(run.started_at)}</dd>
                </div>
                <div>
                  <dt>结束时间</dt>
                  <dd>{dateText(run.finished_at)}</dd>
                </div>
              </dl>
              <pre>{JSON.stringify(run.config, null, 2)}</pre>
            </div>
          </details>
        </>
      )}
      <footer className="lab-footer">
        <span>
          <i /> TinyGPT · 训练观测工作台
        </span>
        <span>
          {autoRefresh
            ? "指标每 5 秒更新 · 实验列表每 15 秒更新"
            : "自动更新已暂停"}
        </span>
      </footer>
    </section>
  );
}
