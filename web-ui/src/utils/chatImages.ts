import {
  ACCEPT_IMAGE_TYPES,
  MAX_COMPOSER_IMAGES,
  MAX_IMAGE_BYTES,
  type ComposerImage,
} from '../types/chatImage';

function randomId(): string {
  return `img-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function normalizeMediaType(file: File): string {
  const mt = (file.type || '').toLowerCase();
  if (mt === 'image/jpg') {
    return 'image/jpeg';
  }
  return mt || 'image/png';
}

export function fileToComposerImage(file: File): ComposerImage {
  if (!ACCEPT_IMAGE_TYPES.split(',').includes(normalizeMediaType(file))) {
    throw new Error(`不支持的图片类型: ${file.type || '未知'}`);
  }
  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error(`图片 ${file.name} 超过 5MB 限制`);
  }
  return {
    id: randomId(),
    name: file.name,
    media_type: normalizeMediaType(file),
    preview_url: URL.createObjectURL(file),
    file,
  };
}

export function filesToComposerImages(
  files: FileList | File[],
  existingCount: number,
): ComposerImage[] {
  const list = Array.from(files).filter(
    (f) => f.type.startsWith('image/') || /\.(png|jpe?g|webp|gif)$/i.test(f.name),
  );
  if (list.length === 0) {
    return [];
  }
  const room = MAX_COMPOSER_IMAGES - existingCount;
  if (room <= 0) {
    throw new Error(`最多上传 ${MAX_COMPOSER_IMAGES} 张图片`);
  }
  const picked = list.slice(0, room);
  const out = picked.map((file) => fileToComposerImage(file));
  if (list.length > room) {
    throw new Error(`最多还可添加 ${room} 张图片`);
  }
  return out;
}

export function revokeComposerImagePreview(img: ComposerImage): void {
  URL.revokeObjectURL(img.preview_url);
}

export function isImageDragEvent(e: React.DragEvent): boolean {
  const types = Array.from(e.dataTransfer?.types ?? []);
  return types.includes('Files');
}
