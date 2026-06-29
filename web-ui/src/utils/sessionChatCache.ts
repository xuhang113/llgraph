import type { ChatMessage } from '../components/console/ChatThread';

const cache = new Map<string, ChatMessage[]>();

function cacheKey(slug: string, threadId: string): string {
  return `${slug}:${threadId}`;
}

export function getSessionChatCache(slug: string, threadId: string): ChatMessage[] | undefined {
  if (!slug || !threadId) {
    return undefined;
  }
  const hit = cache.get(cacheKey(slug, threadId));
  return hit?.length ? hit : undefined;
}

export function setSessionChatCache(
  slug: string,
  threadId: string,
  messages: ChatMessage[],
): void {
  if (!slug || !threadId || messages.length === 0) {
    return;
  }
  cache.set(cacheKey(slug, threadId), messages);
}

export function clearSessionChatCache(slug: string, threadId: string): void {
  if (!slug || !threadId) {
    return;
  }
  cache.delete(cacheKey(slug, threadId));
}
