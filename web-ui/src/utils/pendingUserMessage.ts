const PREFIX = 'llgraph-pending-user:';

export function pendingUserMessageKey(slug: string, threadId: string): string {
  return `${PREFIX}${slug}:${threadId}`;
}

export function savePendingUserMessage(slug: string, threadId: string, text: string): void {
  const t = text.trim();
  if (!slug || !threadId || !t) {
    return;
  }
  try {
    sessionStorage.setItem(pendingUserMessageKey(slug, threadId), t);
  } catch {
    /* ignore */
  }
}

export function loadPendingUserMessage(slug: string, threadId: string): string | null {
  if (!slug || !threadId) {
    return null;
  }
  try {
    return sessionStorage.getItem(pendingUserMessageKey(slug, threadId));
  } catch {
    return null;
  }
}

export function clearPendingUserMessage(slug: string, threadId: string): void {
  if (!slug || !threadId) {
    return;
  }
  try {
    sessionStorage.removeItem(pendingUserMessageKey(slug, threadId));
  } catch {
    /* ignore */
  }
}
