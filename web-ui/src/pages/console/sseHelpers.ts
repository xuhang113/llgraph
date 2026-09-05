import type { TreeNode } from '../../api/client';
import { POST_STREAM_ACTIVE_MS } from './constants';

/** cli-xxx:explore:id / cli-xxx:subagent:id → 父 thread */
export function parentThreadFromSubThread(subThread: string): string {
  for (const marker of [':explore:', ':subagent:'] as const) {
    const idx = subThread.indexOf(marker);
    if (idx >= 0) {
      return subThread.slice(0, idx);
    }
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
    const main = parentThreadFromSubThread(sessionThread);
    const viewing =
      selected.thread_id === sessionThread || selected.thread_id === main;
    if (!viewing) {
      return false;
    }
  }
  const candidates = [sessionThread, parentThreadFromSubThread(sessionThread)];
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

/** Agent 内 spawn 的 explore / 通用 subagent */
export function isExploreSubagentEvent(event: Record<string, unknown>): boolean {
  const kind = String(event.subgraph_kind || '');
  if (kind === 'explore' || kind === 'subagent') {
    return true;
  }
  const sub = String(event.sub_thread || '');
  return sub.includes(':explore:') || sub.includes(':subagent:');
}
