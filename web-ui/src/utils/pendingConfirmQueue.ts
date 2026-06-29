/** 待确认项队列（Survey / Plan confirm / Work 步进），按 thread 持久化到 sessionStorage。 */

export type PendingConfirmKind = 'survey' | 'plan_confirm' | 'task_step_confirm';

export interface PendingConfirmItem {
  id: string;
  kind: PendingConfirmKind;
  payload: unknown;
  createdAt: number;
  assistantMessageId?: string;
}

const PREFIX = 'llgraph-pending-confirm:';

export function pendingConfirmQueueKey(slug: string, threadId: string): string {
  return `${PREFIX}${slug}:${threadId}`;
}

function loadQueue(slug: string, threadId: string): PendingConfirmItem[] {
  if (!slug || !threadId) {
    return [];
  }
  try {
    const raw = sessionStorage.getItem(pendingConfirmQueueKey(slug, threadId));
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(
      (item): item is PendingConfirmItem =>
        Boolean(item)
        && typeof item === 'object'
        && typeof (item as PendingConfirmItem).id === 'string'
        && typeof (item as PendingConfirmItem).kind === 'string'
        && typeof (item as PendingConfirmItem).createdAt === 'number',
    );
  } catch {
    return [];
  }
}

function saveQueue(slug: string, threadId: string, queue: PendingConfirmItem[]): void {
  if (!slug || !threadId) {
    return;
  }
  try {
    if (queue.length === 0) {
      sessionStorage.removeItem(pendingConfirmQueueKey(slug, threadId));
      return;
    }
    sessionStorage.setItem(pendingConfirmQueueKey(slug, threadId), JSON.stringify(queue));
  } catch {
    /* ignore */
  }
}

function newConfirmId(kind: PendingConfirmKind): string {
  return `${kind}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/** 入队；plan_confirm / task_step_confirm 同 kind 只保留最新一项。 */
export function enqueueConfirm(
  slug: string,
  threadId: string,
  item: Omit<PendingConfirmItem, 'id' | 'createdAt'> & { id?: string; createdAt?: number },
): PendingConfirmItem {
  const entry: PendingConfirmItem = {
    id: item.id || newConfirmId(item.kind),
    kind: item.kind,
    payload: item.payload,
    createdAt: item.createdAt ?? Date.now(),
    assistantMessageId: item.assistantMessageId,
  };
  const queue = loadQueue(slug, threadId);
  let next: PendingConfirmItem[];
  if (entry.kind === 'plan_confirm' || entry.kind === 'task_step_confirm') {
    next = [...queue.filter((q) => q.kind !== entry.kind), entry];
  } else {
    next = [...queue, entry];
  }
  saveQueue(slug, threadId, next);
  return entry;
}

export function peekConfirmQueue(slug: string, threadId: string): PendingConfirmItem[] {
  return loadQueue(slug, threadId);
}

export function peekConfirmHead(slug: string, threadId: string): PendingConfirmItem | null {
  const queue = loadQueue(slug, threadId);
  return queue[0] ?? null;
}

export function dequeueConfirm(slug: string, threadId: string, id: string): void {
  const queue = loadQueue(slug, threadId);
  saveQueue(
    slug,
    threadId,
    queue.filter((item) => item.id !== id),
  );
}

export function dequeueConfirmHead(slug: string, threadId: string): PendingConfirmItem | null {
  const queue = loadQueue(slug, threadId);
  if (queue.length === 0) {
    return null;
  }
  const [head, ...rest] = queue;
  saveQueue(slug, threadId, rest);
  return head;
}

export function clearConfirmQueue(
  slug: string,
  threadId: string,
  kind?: PendingConfirmKind,
): void {
  if (!kind) {
    saveQueue(slug, threadId, []);
    return;
  }
  saveQueue(
    slug,
    threadId,
    loadQueue(slug, threadId).filter((item) => item.kind !== kind),
  );
}

export function countPendingConfirms(slug: string, threadId: string): number {
  return loadQueue(slug, threadId).length;
}

export function hasPendingKind(
  slug: string,
  threadId: string,
  kind: PendingConfirmKind,
): boolean {
  return loadQueue(slug, threadId).some((item) => item.kind === kind);
}
