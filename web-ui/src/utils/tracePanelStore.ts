import type { TraceStep } from '../types/trace';
import { mergeTraceStepsUnique, stepsToPanelLogLines } from '../types/trace';

const PREFIX = 'llgraph-trace:';

interface CachedTrace {
  log_lines: string[];
  steps: TraceStep[];
}

export function tracePanelCacheKey(slug: string, threadId: string): string {
  return `${PREFIX}${slug}:${threadId}`;
}

export function saveTracePanelCache(
  slug: string,
  threadId: string,
  logLines: string[],
  steps: TraceStep[],
): void {
  if (!slug || !threadId) {
    return;
  }
  if (logLines.length === 0 && steps.length === 0) {
    return;
  }
  try {
    const payload: CachedTrace = { log_lines: logLines, steps };
    sessionStorage.setItem(tracePanelCacheKey(slug, threadId), JSON.stringify(payload));
  } catch {
    /* ignore quota */
  }
}

/** 向缓存追加一行 trace（后台 SSE 落盘用）。 */
export function appendTracePanelCacheLine(
  slug: string,
  threadId: string,
  line: string,
): void {
  if (!slug || !threadId || !line.trim()) {
    return;
  }
  const cached = loadTracePanelCache(slug, threadId) || { log_lines: [], steps: [] };
  const trimmed = line.trim();
  const last = cached.log_lines[cached.log_lines.length - 1];
  if (last && last.trim() === trimmed) {
    return;
  }
  saveTracePanelCache(slug, threadId, [...cached.log_lines, line], cached.steps);
}

/** 合并一轮 turn_done 的 steps / 额外行到缓存。 */
export function mergeTracePanelCacheTurn(
  slug: string,
  threadId: string,
  turnSteps: TraceStep[],
  extraLines: string[] = [],
): void {
  if (!slug || !threadId) {
    return;
  }
  if (turnSteps.length === 0 && extraLines.length === 0) {
    return;
  }
  const cached = loadTracePanelCache(slug, threadId) || { log_lines: [], steps: [] };
  let nextSteps = cached.steps;
  let newLines = [...cached.log_lines, ...extraLines];
  if (turnSteps.length > 0) {
    const prevIds = new Set(cached.steps.map((s) => s.step_id));
    nextSteps = mergeTraceStepsUnique(cached.steps, turnSteps);
    const added = nextSteps.filter((s) => !prevIds.has(s.step_id));
    if (added.length > 0) {
      newLines = [...newLines, ...stepsToPanelLogLines(added).map((l) => l.text)];
    }
  }
  saveTracePanelCache(slug, threadId, newLines, nextSteps);
}

/** 向缓存追加单个 trace 步骤（后台/切换会话落盘用）。 */
export function appendTracePanelCacheStep(
  slug: string,
  threadId: string,
  step: TraceStep,
): void {
  if (!slug || !threadId) {
    return;
  }
  const cached = loadTracePanelCache(slug, threadId) || { log_lines: [], steps: [] };
  if (cached.steps.some((s) => s.step_id === step.step_id)) {
    return;
  }
  const nextSteps = mergeTraceStepsUnique(cached.steps, [step]);
  const synth = stepsToPanelLogLines([step]).map((l) => l.text);
  saveTracePanelCache(slug, threadId, [...cached.log_lines, ...synth], nextSteps);
}

export function loadTracePanelCache(slug: string, threadId: string): CachedTrace | null {
  if (!slug || !threadId) {
    return null;
  }
  try {
    const raw = sessionStorage.getItem(tracePanelCacheKey(slug, threadId));
    if (!raw) {
      return null;
    }
    const data = JSON.parse(raw) as CachedTrace;
    if (!data || typeof data !== 'object') {
      return null;
    }
    return {
      log_lines: Array.isArray(data.log_lines) ? data.log_lines.map(String) : [],
      steps: Array.isArray(data.steps) ? (data.steps as TraceStep[]) : [],
    };
  } catch {
    return null;
  }
}
