import type { TreeNode } from '../../api/client';
import { POST_STREAM_ACTIVE_MS } from './constants';

/** plan-xxx:planner:v1 / plan-xxx:worker:w1 → plan-xxx */
export function planMainThreadFromSubThread(subThread: string): string {
  const plannerIdx = subThread.indexOf(':planner:');
  if (plannerIdx >= 0) {
    return subThread.slice(0, plannerIdx);
  }
  const workerIdx = subThread.indexOf(':worker:');
  if (workerIdx >= 0) {
    return subThread.slice(0, workerIdx);
  }
  return subThread;
}

export function shouldSuppressSessionTrace(
  sessionThread: string,
  activeStreams: Map<string, AbortController>,
  lastEventAt: Map<string, number>,
  selected: TreeNode | null,
): boolean {
  if (sessionThread.startsWith('cli-')) {
    return false;
  }
  if (selected) {
    const main = planMainThreadFromSubThread(sessionThread);
    const viewing =
      selected.thread_id === sessionThread ||
      selected.thread_id === main ||
      (selected.kind === 'worker' && selected.thread_id === sessionThread);
    if (!viewing) {
      return false;
    }
  }
  const candidates = [sessionThread, planMainThreadFromSubThread(sessionThread)];
  for (const key of candidates) {
    if (!activeStreams.has(key)) {
      continue;
    }
    const last = lastEventAt.get(key) ?? 0;
    if (Date.now() - last < POST_STREAM_ACTIVE_MS) {
      return true;
    }
  }
  return false;
}

export function isSubagentWorkerEvent(event: Record<string, unknown>): boolean {
  const kind = String(event.subgraph_kind || '');
  if (kind === 'worker') {
    return true;
  }
  return Boolean(event.task_id);
}

export function isPlannerSubagentEvent(event: Record<string, unknown>): boolean {
  return String(event.subgraph_kind || '') === 'planner';
}

export function eventMatchesWorkerTask(
  event: Record<string, unknown>,
  workerThread: string,
  taskId: string,
): boolean {
  const sub = String(event.sub_thread || '');
  if (sub && sub === workerThread) {
    return true;
  }
  const tid = String(event.task_id || '');
  return Boolean(tid && tid === taskId);
}
