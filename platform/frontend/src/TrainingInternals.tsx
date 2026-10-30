import { useState } from "react";
import {
  ArrowRight,
  Box,
  BrainCircuit,
  Check,
  Layers,
  Radio,
  Terminal,
  Zap,
} from "lucide-react";
import { buildSeries } from "./training-data";
import type { Diagnostic, TrainingMetric, TrainingRun } from "./training-data";
import { number, scientific, token } from "./training-format";
import { Stat, Plot } from "./TrainingCharts";

const stages = [
  {
    name: "输入与分词",
    code: "TOKENIZE",
    description:
      "原始文本按字符映射为 Token ID。训练窗口右移一位得到目标序列，每个位置学习预测下一个 Token。",
    icon: Terminal,
  },
  {
    name: "词嵌入与位置",
    code: "EMBEDDING",
    description:
      "Token embedding 与 position embedding 相加，把离散字符转换为带位置信息的向量；嵌入参数随梯度更新。",
    icon: Box,
  },
  {
    name: "Transformer",
    code: "ATTENTION + FFN",
    description:
      "每层先做 LayerNorm 与因果自注意力，再做 LayerNorm 与前馈网络，两次残差连接传递信息。注意力只允许读取当前及之前的位置。",
    icon: Layers,
  },
  {
    name: "预测与损失",
    code: "LOGITS → LOSS",
    description:
      "最终归一化后，语言模型头输出词表中每个 Token 的分数。交叉熵度量真实下一个 Token 的负对数概率，Loss 越低表示当前批次的预测误差越小。",
    icon: BrainCircuit,
  },
  {
    name: "反向传播与更新",
    code: "BACKWARD → ADAMW",
    description:
      "反向传播计算各参数的梯度，AdamW 结合学习率、动量与权重衰减更新参数。更新幅度由更新前后参数的真实差值计算。",
    icon: Zap,
  },
];

export default function Internals({
  metrics,
  current,
  run,
}: {
  metrics: TrainingMetric[];
  current: TrainingMetric;
  run: TrainingRun;
}) {
  const [stage, setStage] = useState(2);
  const [focus, setFocus] = useState("");
  const snapshot = [...metrics]
    .reverse()
    .find((m) => m.global_step <= current.global_step && m.diagnostics);
  const diagnostic = snapshot?.diagnostics;
  const layer =
    diagnostic?.layers.find((item) => item.name === focus) ??
    diagnostic?.layers[0];
  const config = run.config;
  return (
    <div className="lab-internals">
      <section className="lab-card">
        <div className="lab-card-heading">
          <div>
            <h3>一次训练，模型里发生了什么？</h3>
            <p>点击阶段查看机制；下方诊断展示实际训练采样。</p>
          </div>
          <BrainCircuit size={20} />
        </div>
        <div className="lab-flow">
          {stages.map((item, i) => (
            <button
              type="button"
              key={item.code}
              className={stage === i ? "selected" : ""}
              aria-pressed={stage === i}
              onClick={() => setStage(i)}
            >
              <item.icon size={21} />
              <small>
                0{i + 1} / {item.code}
              </small>
              <strong>{item.name}</strong>
              {i < stages.length - 1 && (
                <ArrowRight className="flow-arrow" size={14} />
              )}
            </button>
          ))}
        </div>
        <div className="lab-stage-detail">
          <span>0{stage + 1}</span>
          <div>
            <strong>{stages[stage].name}</strong>
            <p>{stages[stage].description}</p>
          </div>
        </div>
        <div className="lab-shapes">
          <span>
            输入 [B, T]{" "}
            <b>
              [{number(config.batch_size, 0)},{" "}
              {number(config.context_length, 0)}]
            </b>
          </span>
          <ArrowRight size={13} />
          <span>
            隐藏状态 [B, T, D] <b>D = {number(config.d_model, 0)}</b>
          </span>
          <ArrowRight size={13} />
          <span>
            输出 [B, T, V] <b>V = {number(config.vocab_size, 0)}</b>
          </span>
        </div>
        <p className="lab-note">
          B 为配置批大小，最后一批可能更小；T
          为上下文长度。以上是训练机制说明，不代表浏览器在实时执行这些计算。
        </p>
      </section>
      {!diagnostic ? (
        <section className="lab-empty lab-card">
          <Layers size={32} />
          <h3>这个实验还没有模型内部诊断</h3>
          <p>
            旧实验保留原有训练指标。新训练开启采样后，可查看梯度、参数更新、激活、预测概率和注意力。
          </p>
          <code>
            python scripts/train.py --max-steps 30 --api-base-url
            http://127.0.0.1:8080 --diagnostics-interval 10
          </code>
          <p>
            诊断默认关闭，开启后会保存首个样本的短文本片段；请使用适合展示的训练数据。
          </p>
        </section>
      ) : (
        <>
          <div className="lab-snapshot-caption">
            <span>
              <Radio size={14} /> 真实诊断快照 · STEP {snapshot.global_step}
            </span>
            <small>
              {snapshot.global_step === current.global_step
                ? "当前回看步骤已采样"
                : `当前回看 Step ${current.global_step} 未采样，展示之前最近一次快照`}
            </small>
          </div>
          <div className="lab-stats internals-stats">
            <Stat
              label="全模型梯度 L2 范数"
              value={scientific(diagnostic.gradient_norm)}
              detail="反向传播后、更新前的梯度"
            />
            <Stat
              label="参数更新 L2 范数"
              value={scientific(diagnostic.update_norm)}
              detail="本次更新前后参数差值"
              accent
            />
            <Stat
              label="批次 Token 准确率"
              value={
                diagnostic.token_accuracy === null
                  ? "—"
                  : `${number(diagnostic.token_accuracy * 100, 1)}%`
              }
              detail="更新前，本批次全部目标位置"
            />
            <Stat
              label="预测分布熵"
              value={number(diagnostic.prediction_entropy)}
              detail="首个样本最后位置 · nats"
            />
          </div>
          <section className="lab-card">
            <div className="lab-card-heading">
              <div>
                <h3>逐层观察</h3>
                <p>选择一层比较激活、梯度与实际更新量，范数均为 L2。</p>
              </div>
              <Layers size={18} />
            </div>
            <div className="lab-layer-layout">
              <div className="lab-layer-list">
                {diagnostic.layers.map((item) => (
                  <button
                    type="button"
                    key={item.name}
                    onClick={() => setFocus(item.name)}
                    aria-pressed={item.name === layer?.name}
                    className={item.name === layer?.name ? "selected" : ""}
                  >
                    <span>
                      {item.name.startsWith("blocks.")
                        ? `Transformer ${Number(item.name.split(".")[1]) + 1}`
                        : item.name}
                    </span>
                    <code>{scientific(item.gradient_norm)}</code>
                    <i
                      style={{
                        width: `${Math.min(100, ((item.gradient_norm ?? 0) / Math.max(...diagnostic.layers.map((l) => l.gradient_norm ?? 0), 1e-12)) * 100)}%`,
                      }}
                    />
                  </button>
                ))}
              </div>
              {layer && (
                <div className="lab-layer-detail">
                  <span className="lab-kicker">SELECTED LAYER</span>
                  <h4>{layer.name}</h4>
                  <dl>
                    <div>
                      <dt>参数数量</dt>
                      <dd>{number(layer.parameter_count, 0)}</dd>
                    </div>
                    <div>
                      <dt>激活均值</dt>
                      <dd>{number(layer.activation_mean, 5)}</dd>
                    </div>
                    <div>
                      <dt>激活标准差</dt>
                      <dd>{number(layer.activation_std, 5)}</dd>
                    </div>
                    <div>
                      <dt>梯度范数</dt>
                      <dd>{scientific(layer.gradient_norm)}</dd>
                    </div>
                    <div>
                      <dt>更新后参数范数</dt>
                      <dd>{scientific(layer.parameter_norm)}</dd>
                    </div>
                    <div>
                      <dt>本次参数更新范数</dt>
                      <dd>{scientific(layer.update_norm)}</dd>
                    </div>
                  </dl>
                  <p>
                    激活来自这次前向传播；参数范数在 AdamW
                    更新后采集。不同层参数规模不同，不能只凭范数大小判断学习好坏。
                  </p>
                </div>
              )}
            </div>
          </section>
          <div className="lab-internal-grid">
            <section className="lab-card">
              <div className="lab-card-heading">
                <div>
                  <h3>下一个字符，模型会选谁？</h3>
                  <p>首个样本最后位置 · 更新前 Top 5 概率</p>
                </div>
                <BrainCircuit size={18} />
              </div>
              {diagnostic.prediction ? (
                <>
                  <div className="lab-context">
                    <small>输入上下文</small>
                    <code>{diagnostic.prediction.context}</code>
                    <span>
                      真实下一个 Token{" "}
                      <b>{token(diagnostic.prediction.target)}</b>
                    </span>
                  </div>
                  <div className="lab-probabilities">
                    {diagnostic.prediction.top_tokens.map((item, i) => (
                      <div
                        key={i}
                        className={
                          item.token === diagnostic.prediction?.target
                            ? "correct"
                            : ""
                        }
                      >
                        <code>{token(item.token)}</code>
                        <div>
                          <i style={{ width: `${item.probability * 100}%` }} />
                        </div>
                        <span>{number(item.probability * 100, 2)}%</span>
                        {item.token === diagnostic.prediction?.target && (
                          <Check size={13} />
                        )}
                      </div>
                    ))}
                  </div>
                  <p className="lab-note">
                    高亮为真实目标；Top 5 概率不一定合计为
                    100%。不同步骤使用的训练样本可能不同。
                  </p>
                </>
              ) : (
                <p className="lab-note">本次快照没有预测概率。</p>
              )}
            </section>
            <Attention key={snapshot.global_step} diagnostic={diagnostic} />
          </div>
          <Plot
            title="模型更新随训练如何变化"
            subtitle="仅显示采样步骤，不对缺失步骤补造数值。"
            data={buildSeries(
              metrics.filter((m) => m.diagnostics),
              1,
            ).map((m) => ({
              ...m,
              gradient_norm: m.diagnostics!.gradient_norm,
              update_norm: m.diagnostics!.update_norm,
            }))}
            lines={[
              { key: "gradient_norm", name: "梯度 L2", color: "#76d7be" },
              { key: "update_norm", name: "更新 L2", color: "#f0bd7b" },
            ]}
            selected={snapshot.global_step}
            compact
          />
        </>
      )}
    </div>
  );
}

function Attention({ diagnostic }: { diagnostic: Diagnostic }) {
  const [detail, setDetail] = useState("");
  const attention = diagnostic.attention;
  return (
    <section className="lab-card">
      <div className="lab-card-heading">
        <div>
          <h3>注意力看向哪里</h3>
          <p>
            {attention
              ? `Transformer ${attention.layer + 1} · Head ${attention.head + 1} · 首个样本`
              : "本次快照未采集注意力"}
          </p>
        </div>
        <Box size={18} />
      </div>
      {attention && (
        <>
          <div className="lab-attention-scroll">
            <table
              className="lab-attention"
              aria-label="因果注意力权重，行为查询位置，列为被关注位置"
            >
              <thead>
                <tr>
                  <th>Q / K</th>
                  {attention.tokens.map((value, i) => (
                    <th key={i} title={`位置 ${i + 1}`}>
                      {token(value)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {attention.weights.map((row, q) => (
                  <tr key={q}>
                    <th title={`查询位置 ${q + 1}`}>
                      {token(attention.tokens[q])}
                    </th>
                    {row.map((weight, k) => (
                      <td key={k}>
                        <button
                          type="button"
                          className={k > q ? "masked" : ""}
                          style={
                            k <= q && weight !== null
                              ? {
                                  background: `rgba(118,215,190,${0.06 + weight * 0.94})`,
                                  color: weight > 0.4 ? "#061a18" : "#c7e7e0",
                                }
                              : undefined
                          }
                          aria-label={`位置 ${q + 1} 关注位置 ${k + 1}，${k > q ? "未来位置已屏蔽" : `权重 ${number(weight, 4)}`}`}
                          title={
                            k > q ? "未来位置：因果掩码" : number(weight, 4)
                          }
                          onClick={() =>
                            setDetail(
                              `位置 ${q + 1}「${token(attention.tokens[q])}」 → 位置 ${k + 1}「${token(attention.tokens[k])}」：${k > q ? "未来位置已屏蔽" : number(weight, 5)}`,
                            )
                          }
                        >
                          {row.length <= 8 && k <= q ? number(weight, 2) : ""}
                        </button>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="lab-heat-legend">
            <span>低</span>
            <i />
            <span>高</span>
            <small>行 → 查询 Token，列 → 被关注 Token</small>
          </div>
          <p className="lab-note" aria-live="polite">
            {detail ||
              "点击单元格查看权重。展示第一个注意力层、第一头、最多 16 个位置，取自 dropout 前权重。"}
          </p>
        </>
      )}
    </section>
  );
}
