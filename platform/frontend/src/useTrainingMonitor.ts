import { useEffect, useState } from "react";
import { parseMetrics, parseMonitor } from "./training-data";
import type { TrainingMetric, TrainingMonitor } from "./training-data";

export default function useTrainingMonitor(runId: number | null, step: number | null, autoRefresh: boolean, revision: number) {
  const [result, setResult] = useState<{ runId: number; monitor: TrainingMonitor | null; diagnostics: TrainingMetric[]; error: string } | null>(null);
  useEffect(() => {
    if (runId === null) return;
    const id = runId;
    const scope = new AbortController();
    let busy = false;
    async function load() {
      if (busy) return;
      busy = true;
      try {
        const options = { cache: "no-store" as RequestCache, signal: AbortSignal.any([scope.signal, AbortSignal.timeout(12000)]) };
        const [monitorResponse, diagnosticResponse] = await Promise.all([
          fetch(`/api/training-runs/${id}/monitor`, options),
          fetch(`/api/training-runs/${id}/diagnostics?limit=1000${step === null ? "" : `&at_step=${step}`}`, options),
        ]);
        if (!monitorResponse.ok || !diagnosticResponse.ok) throw new Error(`HTTP ${!monitorResponse.ok ? monitorResponse.status : diagnosticResponse.status}`);
        const monitor = parseMonitor(await monitorResponse.json(), id);
        const payload = await diagnosticResponse.json();
        if (payload.run_id !== id) throw new Error("诊断实验编号不匹配");
        const diagnostics = parseMetrics(payload);
        if (!scope.signal.aborted) setResult({ runId: id, monitor, diagnostics, error: "" });
      } catch (error) {
        if (!scope.signal.aborted) setResult((current) => ({
          runId: id, monitor: current?.runId === id ? current.monitor : null,
          diagnostics: current?.runId === id ? current.diagnostics.filter((m) => step === null || m.global_step <= step) : [],
          error: error instanceof Error ? error.message : "监控查询失败",
        }));
      } finally { busy = false; }
    }
    void load();
    const timer = autoRefresh ? window.setInterval(() => void load(), 5000) : null;
    return () => { scope.abort(); if (timer !== null) window.clearInterval(timer); };
  }, [runId, step, autoRefresh, revision]);
  return result?.runId === runId ? result : null;
}
