import type { ChatMessage } from '../components/console/ChatThread';
import type { PendingConfirmItem } from './pendingConfirmQueue';

/** Agent-safe stub：Plan / Survey 确认 UI 已移除。 */
export type PendingConfirmUiSetters = Record<string, never>;

export function applyPendingConfirmHead(
  _slug: string,
  _threadId: string,
  _setters?: PendingConfirmUiSetters,
): PendingConfirmItem | null {
  return null;
}

export async function restorePendingConfirmsFromHistory(_opts: {
  slug: string;
  threadId: string;
  messages?: ChatMessage[];
}): Promise<void> {}
