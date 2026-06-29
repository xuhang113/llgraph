import type { ComposerImage } from '../types/chatImage';

const cache = new Map<string, ComposerImage[]>();

function cacheKey(slug: string, threadId: string): string {
  return `${slug}:${threadId}`;
}

export function getComposerImagesCache(slug: string, threadId: string): ComposerImage[] {
  if (!slug || !threadId) {
    return [];
  }
  const hit = cache.get(cacheKey(slug, threadId));
  return hit ? [...hit] : [];
}

export function setComposerImagesCache(
  slug: string,
  threadId: string,
  images: ComposerImage[],
): void {
  if (!slug || !threadId) {
    return;
  }
  if (images.length === 0) {
    cache.delete(cacheKey(slug, threadId));
    return;
  }
  cache.set(cacheKey(slug, threadId), [...images]);
}

export function clearComposerImagesCache(slug: string, threadId: string): void {
  if (!slug || !threadId) {
    return;
  }
  cache.delete(cacheKey(slug, threadId));
}
