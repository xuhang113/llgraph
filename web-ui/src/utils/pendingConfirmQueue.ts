/** Agent-safe stub：Plan / Survey 确认队列已移除。 */

export type PendingConfirmKind = never;

export interface PendingConfirmItem {
  id: string;
  kind: string;
  payload: unknown;
  createdAt: number;
  assistantMessageId?: string;
}

export function pendingConfirmQueueKey(slug: string, threadId: string): string {
  return `llgraph-pending-confirm:${slug}:${threadId}`;
}

export function enqueueConfirm(
  _slug: string,
  _threadId: string,
  item: Omit<PendingConfirmItem, 'id' | 'createdAt'> & { id?: string; createdAt?: number },
): PendingConfirmItem {
  return {
    id: item.id || `stub-${Date.now()}`,
    kind: item.kind,
    payload: item.payload,
    createdAt: item.createdAt ?? Date.now(),
    assistantMessageId: item.assistantMessageId,
  };
}

export function peekConfirmQueue(_slug: string, _threadId: string): PendingConfirmItem[] {
  return [];
}

export function peekConfirmHead(_slug: string, _threadId: string): PendingConfirmItem | null {
  return null;
}

export function dequeueConfirm(_slug: string, _threadId: string, _id: string): void {}

export function dequeueConfirmHead(_slug: string, _threadId: string): PendingConfirmItem | null {
  return null;
}

export function clearConfirmQueue(_slug: string, _threadId: string, _kind?: string): void {}

export function countPendingConfirms(_slug: string, _threadId: string): number {
  return 0;
}

export function hasPendingKind(_slug: string, _threadId: string, _kind: string): boolean {
  return false;
}
