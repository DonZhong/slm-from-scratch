import { useEffect, useRef, useState } from "react";
import { parseRuns, parseMetrics, mergeMetrics } from "./training-data";
import type { TrainingMetric, TrainingRun } from "./training-data";

export default function useTrainingData(
  autoRefresh: boolean,
  revision: number,
) {
  const historyScope = useRef<AbortController | null>(null);
  const [historyState, setHistoryState] = useState<{ runId: number | null; loading: boolean; error: string }>({ runId: null, loading: false, error: "" });
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(() => {
    const id = Number(new URLSearchParams(window.location.search).get("run"));
    return Number.isSafeInteger(id) && id > 0 ? id : null;
  });
  const [runsState, setRunsState] = useState({ loading: true, error: "" });
  const [result, setResult] = useState<{
    runId: number;
    items: TrainingMetric[];
    error: string;
    loading: boolean;
    updated: Date | null;
  } | null>(null);

  useEffect(() => {
    const scope = new AbortController();
    let busy = false;
    async function load() {
      if (busy) return;
      busy = true;
      try {
        const response = await fetch("/api/training-runs", {
          cache: "no-store",
          signal: AbortSignal.any([scope.signal, AbortSignal.timeout(12000)]),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const next = parseRuns(await response.json());
        if (scope.signal.aborted) return;
        setRuns(next);
        setRunsState({ loading: false, error: "" });
        setSelectedId((current) =>
          next.some((run) => run.id === current)
            ? current
            : (next[0]?.id ?? null),
        );
      } catch (error) {
        if (!scope.signal.aborted)
          setRunsState({
            loading: false,
            error: error instanceof Error ? error.message : "无法获取实验",
          });
      } finally {
        busy = false;
      }
    }
    void load();
    const timer = autoRefresh
      ? window.setInterval(() => void load(), 15000)
      : null;
    return () => {
      scope.abort();
      if (timer !== null) window.clearInterval(timer);
    };
  }, [autoRefresh, revision]);

  useEffect(() => {
    if (selectedId === null) return;
    const runId = selectedId;
    const scope = new AbortController();
    let busy = false;
    async function load() {
      if (busy) return;
      busy = true;
      setResult((current) =>
        current?.runId === runId
          ? { ...current, loading: true }
          : { runId, items: [], error: "", loading: true, updated: null },
      );
      try {
        const response = await fetch(`/api/training-runs/${runId}/metrics`, {
          cache: "no-store",
          signal: AbortSignal.any([scope.signal, AbortSignal.timeout(12000)]),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const items = parseMetrics(await response.json());
        if (!scope.signal.aborted)
          setResult((current) => ({
            runId,
            items: current?.runId === runId ? mergeMetrics(current.items, items) : items,
            error: "", loading: false, updated: new Date(),
          }));
      } catch (error) {
        if (!scope.signal.aborted)
          setResult((current) => ({
            runId,
            items: current?.runId === runId ? current.items : [],
            updated: current?.runId === runId ? current.updated : null,
            loading: false,
            error: error instanceof Error ? error.message : "无法获取指标",
          }));
      } finally {
        busy = false;
      }
    }
    void load();
    const timer = autoRefresh
      ? window.setInterval(() => void load(), 5000)
      : null;
    return () => {
      scope.abort();
      historyScope.current?.abort();
      if (timer !== null) window.clearInterval(timer);
    };
  }, [selectedId, autoRefresh, revision]);

  async function loadEarlier() {
    if (selectedId === null || result?.runId !== selectedId || !result.items.length || (historyState.runId === selectedId && historyState.loading)) return;
    historyScope.current?.abort();
    const scope = new AbortController();
    historyScope.current = scope;
    const runId = selectedId;
    setHistoryState({ runId, loading: true, error: "" });
    try {
      const response = await fetch(`/api/training-runs/${runId}/metrics/history?limit=1000&before_step=${result.items[0].global_step}`, {
        cache: "no-store", signal: AbortSignal.any([scope.signal, AbortSignal.timeout(12000)]),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (payload.run_id !== runId) throw new Error("历史实验编号不匹配");
      const items = parseMetrics(payload);
      if (!scope.signal.aborted) setResult((current) => current?.runId === runId ? { ...current, items: mergeMetrics(items, current.items) } : current);
      if (!scope.signal.aborted) setHistoryState({ runId, loading: false, error: "" });
    } catch (error) {
      if (!scope.signal.aborted) setHistoryState({ runId, loading: false, error: error instanceof Error ? error.message : "历史查询失败" });
    } finally {
      if (historyScope.current === scope) setHistoryState((current) => ({ ...current, loading: false }));
    }
  }

  return {
    loadEarlier, historyState: historyState.runId === selectedId ? historyState : { loading: false, error: "" },
    runs,
    selectedId,
    setSelectedId,
    runsState,
    result: result?.runId === selectedId ? result : null,
  };
}
