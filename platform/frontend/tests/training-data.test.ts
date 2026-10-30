import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildSeries,
  epochSummaries,
  metricsCsv,
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
