import type { TrainingMonitor } from "./training-data";
import { number, dateText } from "./training-format";
import { Stat } from "./TrainingCharts";
const phases: Record<string, string> = { training: "训练更新", validation: "验证评估", checkpoint: "保存 Checkpoint", completed: "训练已完成", failed: "训练异常" };
export default function TrainingMonitorCard({ monitor, error }: { monitor: TrainingMonitor | null; error: string }) {
  if (!monitor) return <div className={error ? "lab-error" : "lab-note"}>{error ? `运行监控读取失败：${error}` : "正在读取运行监控…"}</div>;
  const { summary, runtime, config } = monitor;
  const phase = typeof runtime.phase === "string" ? runtime.phase : "";
  return <section className="lab-card lab-runtime" aria-label="真实运行监控">
    <div className="lab-card-heading"><div><h3>运行监控与全程统计</h3><p>来自训练进程上报和数据库完整记录，不受图表加载范围影响。</p></div></div>
    {error && <p className="lab-error">监控更新失败：{error}，以下保留上次数据。</p>}
    <div className="lab-stats">
      <Stat label="最近上报阶段" value={phases[phase] ?? "旧实验未上报阶段"} detail={`Step ${number(runtime.global_step, 0)} · Epoch ${number(runtime.epoch, 0)} · ${dateText(monitor.updated_at)}`} />
      <Stat label="完整记录 / 内部采样" value={`${number(summary.total_metrics, 0)} / ${number(summary.diagnostics_count, 0)}`} detail={`最新已写入 Step ${number(summary.latest_step, 0)}`} />
      <Stat label="全程最低 Loss" value={number(summary.best_loss, 4)} detail={`首步 ${number(summary.first_loss, 4)} → 末步 ${number(summary.last_loss, 4)}`} />
      <Stat label="全程 Loss 变化" value={summary.loss_change_percent === null ? "—" : `${number(summary.loss_change_percent, 1)}%`} detail={`全程平均吞吐 ${number(summary.average_speed, 1)} tokens/s`} />
    </div>
    <p className="lab-note">训练 / 验证样本：{number(config.train_samples, 0)} / {number(config.val_samples, 0)} · 滑窗步长 {number(config.stride, 0)} · 优化器 {typeof config.optimizer === "string" ? config.optimizer : "未记录"} · 诊断间隔 {number(config.diagnostics_interval, 0)} steps</p>
    {typeof runtime.message === "string" && runtime.message && <p className="lab-error">{runtime.message}</p>}
    {typeof runtime.checkpoint_path === "string" && <p className="lab-note">Checkpoint：{runtime.checkpoint_path} · {number(runtime.checkpoint_size_bytes, 0)} bytes</p>}
    {phase === "completed" && runtime.checkpoint_path === null && <p className="lab-note">本次训练未请求保存 Checkpoint。</p>}
  </section>;
}
