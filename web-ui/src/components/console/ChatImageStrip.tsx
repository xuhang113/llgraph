import { useState } from 'react';
import type { ChatImageAttachment, ComposerImage } from '../../types/chatImage';
import { revokeComposerImagePreview } from '../../utils/chatImages';
import ChatImageLightbox from './ChatImageLightbox';

type ImageItem = ComposerImage | ChatImageAttachment;

function imagePreviewSrc(item: ImageItem): string {
  if ('preview_url' in item && item.preview_url) {
    return item.preview_url;
  }
  if ('url' in item && item.url) {
    return item.url;
  }
  return '';
}

function imageKey(item: ImageItem, index: number): string {
  if ('id' in item && item.id) {
    return item.id;
  }
  const src = imagePreviewSrc(item);
  return `hist-${index}-${src.slice(0, 24)}`;
}

interface Props {
  images: ImageItem[];
  editable?: boolean;
  onRemove?: (id: string) => void;
}

export default function ChatImageStrip({ images, editable = false, onRemove }: Props) {
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  if (images.length === 0) {
    return null;
  }

  return (
    <>
      <div className={`cursor-chat-image-strip${editable ? ' is-editable' : ''}`}>
        {images.map((img, index) => {
          const key = imageKey(img, index);
          const removeId = 'id' in img && img.id ? img.id : key;
          const src = imagePreviewSrc(img);
          return (
            <div key={key} className="cursor-chat-image-item">
              <button
                type="button"
                className="cursor-chat-image-thumb-btn"
                aria-label="放大预览图片"
                disabled={!src}
                onClick={() => {
                  if (src) {
                    setLightboxSrc(src);
                  }
                }}
              >
                <img src={src} alt="" className="cursor-chat-image-thumb" />
              </button>
              {editable && onRemove && (
                <button
                  type="button"
                  className="cursor-chat-image-remove"
                  aria-label="移除图片"
                  onClick={() => {
                    if ('preview_url' in img) {
                      revokeComposerImagePreview(img);
                    }
                    onRemove(removeId);
                  }}
                >
                  ×
                </button>
              )}
            </div>
          );
        })}
      </div>
      {lightboxSrc && (
        <ChatImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}
    </>
  );
}
