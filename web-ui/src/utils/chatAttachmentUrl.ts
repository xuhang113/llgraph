import type { ChatImageAttachment } from '../types/chatImage';
import type { ChatMessage } from '../components/console/ChatThread';

/** 会话附件预览 URL（与后端 attachment_api_path 对齐）。 */
export function sessionAttachmentUrl(
  slug: string,
  threadId: string,
  imageId: string,
): string {
  const id = String(imageId || '').trim();
  if (!slug || !threadId || !id) {
    return '';
  }
  return `/api/workspaces/${encodeURIComponent(slug)}/sessions/${encodeURIComponent(threadId)}/attachments/${encodeURIComponent(id)}`;
}

function needsAttachmentHydration(url: string | undefined): boolean {
  const u = String(url || '').trim();
  return !u || u.startsWith('blob:');
}

/** 用稳定附件 URL 补齐/替换 blob 预览（刷新后 blob 失效）。 */
export function hydrateAttachmentImages(
  images: ChatImageAttachment[] | undefined,
  slug: string,
  threadId: string,
): ChatImageAttachment[] | undefined {
  if (!images?.length) {
    return images;
  }
  let changed = false;
  const next = images.map((img) => {
    const id = String(img.id || '').trim();
    if (!id || !needsAttachmentHydration(img.url)) {
      return img;
    }
    const url = sessionAttachmentUrl(slug, threadId, id);
    if (!url || url === img.url) {
      return img;
    }
    changed = true;
    return { ...img, url };
  });
  return changed ? next : images;
}

/** 历史消息：把用户气泡里的 blob/缺 url 换成附件接口。 */
export function hydrateChatMessageImages(
  messages: ChatMessage[],
  slug: string,
  threadId: string,
): ChatMessage[] {
  if (!slug || !threadId || messages.length === 0) {
    return messages;
  }
  let changed = false;
  const next = messages.map((m) => {
    if (m.role !== 'user') {
      return m;
    }
    const images = hydrateAttachmentImages(m.images, slug, threadId);
    if (images === m.images) {
      return m;
    }
    changed = true;
    return { ...m, images };
  });
  return changed ? next : messages;
}
