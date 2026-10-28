export type Config = Record<string, unknown>;
export type TrainingRun = {
  id: number;
  name: string;
  model_name: string;
  dataset_name: string;
  status: string;
  config: Config;
  created_at: string;
  started_at: string;
  finished_at: string;
};
export type LayerDiagnostic = {
  name: string;
  activation_mean: number | null;
  activation_std: number | null;
  gradient_norm: number | null;
  parameter_norm: number | null;
  update_norm: number | null;
  parameter_count: number | null;
};
export type Diagnostic = {
  layers: LayerDiagnostic[];
  gradient_norm: number | null;
  parameter_norm: number | null;
  update_norm: number | null;
  token_accuracy: number | null;
  prediction_entropy: number | null;
  prediction: {
    context: string;
    target: string;
    top_tokens: { token: string; probability: number }[];
  } | null;
  attention: {
    layer: number;
    head: number;
    tokens: string[];
    weights: (number | null)[][];
  } | null;
};
export type TrainingMetric = {
  global_step: number;
  epoch: number | null;
  train_loss: number | null;
  val_loss: number | null;
  learning_rate: number | null;
  tokens_per_second: number | null;
  diagnostics: Diagnostic | null;
};

export const numeric = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
export const record = (value: unknown): Config =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Config)
    : {};
const text = (value: unknown) => (typeof value === "string" ? value : "");
const positiveInt = (value: unknown) =>
  Number.isSafeInteger(value) && Number(value) > 0;
const nonnegative = (value: unknown) => {
  const n = numeric(value);
  return n !== null && n >= 0 ? n : null;
};
const unit = (value: unknown) => {
  const n = nonnegative(value);
  return n !== null && n <= 1 ? n : null;
};

export function parseDiagnostic(value: unknown): Diagnostic | null {
  const d = record(value);
  if (d.version !== 1) return null;
  const p = record(d.prediction);
  const a = record(d.attention);
  const tokens = Array.isArray(a.tokens) ? a.tokens.slice(0, 16).map(text) : [];
  const weights = Array.isArray(a.weights)
    ? a.weights
        .slice(0, 16)
        .map((row) => (Array.isArray(row) ? row.slice(0, 16).map(unit) : []))
    : [];
  const attentionValid =
    tokens.length > 0 &&
    weights.length === tokens.length &&
    weights.every((row) => row.length === tokens.length);
  return {
    layers: Array.isArray(d.layers)
      ? d.layers.slice(0, 32).map((value) => {
          const layer = record(value);
          return {
            name: text(layer.name),
            activation_mean: numeric(layer.activation_mean),
            activation_std: nonnegative(layer.activation_std),
            gradient_norm: nonnegative(layer.gradient_norm),
            parameter_norm: nonnegative(layer.parameter_norm),
            update_norm: nonnegative(layer.update_norm),
            parameter_count: nonnegative(layer.parameter_count),
          };
        })
      : [],
    gradient_norm: nonnegative(d.gradient_norm),
    parameter_norm: nonnegative(d.parameter_norm),
    update_norm: nonnegative(d.update_norm),
    token_accuracy: unit(d.token_accuracy),
    prediction_entropy: nonnegative(d.prediction_entropy),
    prediction:
      typeof p.context === "string" && typeof p.target === "string"
        ? {
            context: p.context,
            target: p.target,
            top_tokens: Array.isArray(p.top_tokens)
              ? p.top_tokens.slice(0, 5).flatMap((value) => {
                  const token = record(value);
                  const probability = unit(token.probability);
                  return probability === null || typeof token.token !== "string"
                    ? []
                    : [{ token: token.token, probability }];
                })
              : [],
          }
        : null,
    attention: attentionValid
      ? {
          layer: nonnegative(a.layer) ?? 0,
          head: nonnegative(a.head) ?? 0,
          tokens,
          weights,
        }
      : null,
  };
}

export function parseRuns(payload: unknown): TrainingRun[] {
  const data = record(payload);
  if (data.status !== "ok" || !Array.isArray(data.items))
    throw new Error("实验列表格式异常");
  return data.items
    .map(record)
    .filter((item) => positiveInt(item.id))
    .map((item) => ({
      id: Number(item.id),
      name: text(item.name),
      model_name: text(item.model_name),
      dataset_name: text(item.dataset_name),
      status: text(item.status),
      config: record(item.config),
      created_at: text(item.created_at),
      started_at: text(item.started_at),
      finished_at: text(item.finished_at),
    }));
}

export function parseMetrics(payload: unknown): TrainingMetric[] {
  const data = record(payload);
  if (data.status !== "ok" || !Array.isArray(data.items))
    throw new Error("训练指标格式异常");
  const unique = new Map<number, TrainingMetric>();
  for (const item of data.items.map(record)) {
    if (!positiveInt(item.global_step)) continue;
    unique.set(Number(item.global_step), {
      global_step: Number(item.global_step),
      epoch: positiveInt(item.epoch) ? Number(item.epoch) : null,
      train_loss: nonnegative(item.train_loss),
      val_loss: nonnegative(item.val_loss),
      learning_rate: nonnegative(item.learning_rate),
      tokens_per_second: nonnegative(item.tokens_per_second),
      diagnostics: parseDiagnostic(item.diagnostics),
    });
  }
  return [...unique.values()].sort((a, b) => a.global_step - b.global_step);
}

export function buildSeries(metrics: TrainingMetric[], alpha: number) {
  let ema: number | null = null;
  return metrics.map((metric) => {
    if (metric.train_loss !== null)
      ema =
        ema === null
          ? metric.train_loss
          : alpha * metric.train_loss + (1 - alpha) * ema;
    const perplexity =
      metric.train_loss === null ? null : numeric(Math.exp(metric.train_loss));
    return {
      ...metric,
      smoothed_loss: metric.train_loss === null ? null : ema,
      perplexity,
    };
  });
}

export function summarize(metrics: TrainingMetric[]) {
  const losses = metrics.flatMap((m) =>
    m.train_loss === null ? [] : [m.train_loss],
  );
  const speeds = metrics.flatMap((m) =>
    m.tokens_per_second === null ? [] : [m.tokens_per_second],
  );
  const first = losses[0] ?? null;
  const last = losses.at(-1) ?? null;
  const validation = [...metrics].reverse().find((m) => m.val_loss !== null);
  return {
    first,
    last,
    best: losses.length ? Math.min(...losses) : null,
    improvement:
      first !== null && first > 0 && last !== null
        ? ((first - last) / first) * 100
        : null,
    averageSpeed: speeds.length
      ? speeds.reduce((a, b) => a + b, 0) / speeds.length
      : null,
    validation,
  };
}

export function epochSummaries(metrics: TrainingMetric[]) {
  const epochs = new Map<number, TrainingMetric[]>();
  for (const metric of metrics) {
    if (metric.epoch === null) continue;
    const list = epochs.get(metric.epoch) ?? [];
    list.push(metric);
    epochs.set(metric.epoch, list);
  }
  return [...epochs].map(([epoch, items]) => ({
    epoch,
    count: items.length,
    firstStep: items[0].global_step,
    lastStep: items.at(-1)!.global_step,
    ...summarize(items),
  }));
}

export function metricsCsv(metrics: TrainingMetric[]): string {
  const keys = [
    "global_step",
    "epoch",
    "train_loss",
    "val_loss",
    "learning_rate",
    "tokens_per_second",
  ] as const;
  return [
    keys.join(","),
    ...metrics.map((m) => keys.map((key) => m[key] ?? "").join(",")),
  ].join("\n");
}
