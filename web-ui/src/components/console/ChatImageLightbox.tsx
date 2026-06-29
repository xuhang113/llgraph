import { useEffect } from 'react';

interface Props {
  src: string;
  alt?: string;
  onClose: () => void;
}

export default function ChatImageLightbox({ src, alt = '', onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="cursor-chat-image-lightbox"
      role="dialog"
      aria-modal="true"
      aria-label="图片预览"
      onClick={onClose}
    >
      <button
        type="button"
        className="cursor-chat-image-lightbox-close"
        aria-label="关闭预览"
        onClick={onClose}
      >
        ×
      </button>
      <img
        src={src}
        alt={alt}
        className="cursor-chat-image-lightbox-img"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
