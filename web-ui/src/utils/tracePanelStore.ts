import type { TraceStep } from '../types/trace';

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
