const PREFIX = 'llgraph-composer-draft:';

function draftKey(slug: string, threadId: string): string {
  return `${PREFIX}${slug}:${threadId}`;
}

export function saveComposerDraft(slug: string, threadId: string, text: string): void {
  if (!slug || !threadId) {
    return;
  }
  const trimmed = text.trim();
  try {
    if (!trimmed) {
      sessionStorage.removeItem(draftKey(slug, threadId));
    } else {
      sessionStorage.setItem(draftKey(slug, threadId), text);
    }
  } catch {
    /* ignore */
  }
}

export function loadComposerDraft(slug: string, threadId: string): string {
  if (!slug || !threadId) {
    return '';
  }
  try {
    return sessionStorage.getItem(draftKey(slug, threadId)) || '';
  } catch {
    return '';
  }
}

export function clearComposerDraft(slug: string, threadId: string): void {
  if (!slug || !threadId) {
    return;
  }
  try {
    sessionStorage.removeItem(draftKey(slug, threadId));
  } catch {
    /* ignore */
  }
}
