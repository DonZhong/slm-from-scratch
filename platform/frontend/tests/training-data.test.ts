import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildSeries,
  epochSummaries,
  metricsCsv,
  mergeMetrics,
  parseMonitor,
  parseDiagnostic,
  parseMetrics,
  parseRuns,
  summarize,
} from "../src/training-data.ts";

test("sorts and deduplicates steps; preserves missing values and zero without coercion", () => {
  const metrics = parseMetrics({
    status: "ok",
    items: [
      {
        global_step: 3,
        epoch: 1,
        train_loss: 1,
        val_loss: 2,
        tokens_per_second: 0,
      },
      { global_step: 1, epoch: 1, train_loss: 0, val_loss: null },
      { global_step: 2, epoch: 1, train_loss: "0", val_loss: -1 },
      {
        global_step: 3,
        epoch: 2,
        train_loss: 1,
        val_loss: 2,
        tokens_per_second: 0,
      },
      { global_step: "4", train_loss: 1 },
      { global_step: -2 },
    ],
  });
  assert.deepEqual(
    metrics.map((m) => m.global_step),
    [1, 2, 3],
  );
  assert.equal(metrics[1].train_loss, null);
  assert.equal(metrics[1].val_loss, null);
  assert.equal(metrics[2].tokens_per_second, 0);
  assert.equal(summarize(metrics).improvement, null);
  const epochs = epochSummaries(metrics);
  assert.equal(epochs[0].firstStep, 1);
  assert.equal(epochs[0].lastStep, 2);
  assert.equal(epochs[1].last, 1);
});

test("EMA, latest past validation, window statistics and perplexity use real data", () => {
  const metrics = parseMetrics({
    status: "ok",
    items: [
      {
        global_step: 10,
        epoch: 1,
        train_loss: 4,
        val_loss: 5,
        tokens_per_second: 100,
      },
      {
        global_step: 11,
        epoch: 1,
        train_loss: 2,
        val_loss: null,
        tokens_per_second: 200,
      },
      { global_step: 12, epoch: 1, train_loss: 1, val_loss: 2 },
    ],
  });
  const series = buildSeries(metrics, 0.5);
  assert.deepEqual(
    series.map((m) => m.smoothed_loss),
    [4, 3, 2],
  );
  assert.equal(series[2].perplexity, Math.E);
  const summary = summarize(metrics.slice(0, 2));
  assert.equal(summary.improvement, 50);
  assert.equal(summary.averageSpeed, 150);
  assert.equal(summary.validation?.global_step, 10);
  assert.equal(
    buildSeries([{ ...metrics[0], train_loss: 1000 }], 1)[0].perplexity,
    null,
  );
  assert.equal(metricsCsv(metrics).split("\n")[2], "11,1,2,,,200");
});

test("legacy runs and absent or malformed diagnostic samples degrade safely", () => {
  assert.deepEqual(
    parseRuns({ status: "ok", items: [{ id: 1, config: null }] })[0].config,
    {},
  );
  assert.equal(parseDiagnostic(null), null);
  assert.equal(parseDiagnostic({ version: 2 }), null);
  const diagnostic = parseDiagnostic({
    version: 1,
    gradient_norm: "bad",
    layers: [null],
    attention: { tokens: ["a", "b"], weights: [[1]] },
  });
  assert.equal(diagnostic!.gradient_norm, null);
  assert.equal(diagnostic!.attention, null);
  assert.equal(diagnostic!.layers[0].activation_mean, null);
  assert.throws(() => parseMetrics({ status: "error", items: [] }));
});

test("monitor checks experiment identity and preserves nullable whole-run statistics", () => {
  const monitor = parseMonitor({ status: "ok", run_id: 7, config: { stride: 2 }, runtime: { phase: "validation" },
    summary: { total_metrics: 1200, latest_step: 1500, diagnostics_count: 12, best_loss: 0, loss_change_percent: -10, average_speed: null } }, 7);
  assert.equal(monitor.summary.total_metrics, 1200);
  assert.equal(monitor.summary.best_loss, 0);
  assert.equal(monitor.summary.loss_change_percent, -10);
  assert.equal(monitor.summary.average_speed, null);
  assert.throws(() => parseMonitor({ status: "ok", run_id: 8 }, 7));
});
test("history merge retains earlier pages, replaces latest updates and keeps sampled tensors", () => {
  const early = parseMetrics({ status: "ok", items: [{ global_step: 1, train_loss: 2, diagnostics: { version: 1,
    input_shape: [1, 4], shapes: { embeddings: [1, 4, 8] }, input_ids: [1, 2], target_ids: [2, 3], pipeline: { backward: 2, optimizer: -1 } } }] });
  const later = parseMetrics({ status: "ok", items: [{ global_step: 1001, train_loss: 1 }] });
  const updated = parseMetrics({ status: "ok", items: [{ global_step: 1001, train_loss: 0.5 }] });
  const merged = mergeMetrics(early, later, updated);
  assert.deepEqual(merged.map((m) => m.global_step), [1, 1001]);
  assert.equal(merged[1].train_loss, 0.5);
  assert.deepEqual(merged[0].diagnostics!.shapes.embeddings, [1, 4, 8]);
  assert.equal(merged[0].diagnostics!.pipeline.optimizer, null);
});
