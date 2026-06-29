import { useEffect, useMemo, useRef, useState } from 'react';
import type { SlashCatalogItem } from '../../api/client';
import type { ComposerImage } from '../../types/chatImage';
import { ACCEPT_IMAGE_TYPES, IMAGE_ATTACH_TOOLTIP } from '../../types/chatImage';
import { filesToComposerImages, isImageDragEvent } from '../../utils/chatImages';
import { filterSlashCatalog, parseSlashPartial } from '../../utils/slashCatalog';
import ChatImageStrip from './ChatImageStrip';
import SlashCompletionMenu from './SlashCompletionMenu';

interface Props {
  value: string;
  onChange: (v: string) => void;
  images: ComposerImage[];
  onImagesChange: (images: ComposerImage[]) => void;
  onSend: () => void;
  onStop?: () => void;
  busy?: boolean;
  disabled?: boolean;
  placeholder: string;
  slashCatalog?: SlashCatalogItem[];
  embedded?: boolean;
  stopInStatusBar?: boolean;
}

const COMPOSER_MIN_HEIGHT = 44;
const COMPOSER_MAX_HEIGHT = 240;

export default function ChatComposer({
  value,
  onChange,
  images,
  onImagesChange,
  onSend,
  onStop,
  busy = false,
  disabled = false,
  placeholder,
  slashCatalog = [],
  embedded = false,
  stopInStatusBar = false,
}: Props) {
  const inputDisabled = disabled && !busy;
  const composingRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [cursor, setCursor] = useState(0);
  const [menuOpen, setMenuOpen] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [attachHint, setAttachHint] = useState('');

  const canSend = !disabled && (value.trim().length > 0 || images.length > 0);

  const syncTextareaHeight = () => {
    const el = textareaRef.current;
    if (!el) {
      return;
    }
    el.style.height = 'auto';
    const next = Math.min(
      Math.max(el.scrollHeight, COMPOSER_MIN_HEIGHT),
      COMPOSER_MAX_HEIGHT,
    );
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > COMPOSER_MAX_HEIGHT ? 'auto' : 'hidden';
  };

  useEffect(() => {
    syncTextareaHeight();
  }, [value]);

  const partial = useMemo(
    () => parseSlashPartial(value, cursor),
    [value, cursor],
  );

  const suggestions = useMemo(() => {
    if (partial === null || slashCatalog.length === 0) {
      return [];
    }
    return filterSlashCatalog(slashCatalog, partial);
  }, [partial, slashCatalog]);

  const showMenu = menuOpen && partial !== null && suggestions.length > 0;

  useEffect(() => {
    setActiveIndex(0);
  }, [partial, suggestions.length]);

  const syncCursor = () => {
    const el = textareaRef.current;
    if (el) {
      setCursor(el.selectionStart ?? value.length);
    }
  };

  const applySuggestion = (item: SlashCatalogItem) => {
    onChange(item.insert_text);
    setMenuOpen(false);
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (el) {
        el.focus();
        const len = el.value.length;
        el.setSelectionRange(len, len);
        setCursor(len);
      }
    });
  };

  const showAttachError = (message: string) => {
    setAttachHint(message);
    window.setTimeout(() => setAttachHint(''), 3200);
  };

  const addFiles = (files: FileList | File[] | null | undefined) => {
    if (!files || inputDisabled) {
      return;
    }
    try {
      const next = filesToComposerImages(files, images.length);
      if (next.length > 0) {
        onImagesChange([...images, ...next]);
      }
    } catch (err) {
      showAttachError(err instanceof Error ? err.message : '添加图片失败');
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items) {
      return;
    }
    const imageFiles: File[] = [];
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          imageFiles.push(file);
        }
      }
    }
    if (imageFiles.length > 0) {
      e.preventDefault();
      void addFiles(imageFiles);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (!isImageDragEvent(e)) {
      return;
    }
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    if (e.currentTarget.contains(e.relatedTarget as Node)) {
      return;
    }
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    if (!isImageDragEvent(e)) {
      return;
    }
    e.preventDefault();
    setDragOver(false);
    void addFiles(e.dataTransfer.files);
  };

  const hintText = attachHint
    ? attachHint
    : showMenu
      ? '↑↓ 选择 · Tab/Enter 补全 · Esc 关闭'
      : busy
        ? '生成中…'
        : 'Enter 发送 · Shift+Enter 换行 · / 命令 · 可拖拽/粘贴图片';

  const inner = (
    <>
      {showMenu && (
        <SlashCompletionMenu
          items={suggestions}
          activeIndex={activeIndex}
          onSelect={applySuggestion}
          onHover={setActiveIndex}
        />
      )}
      <ChatImageStrip
        images={images}
        editable
        onRemove={(id) => onImagesChange(images.filter((img) => img.id !== id))}
      />
      <textarea
        ref={textareaRef}
        className="cursor-composer-input"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setMenuOpen(true);
          setCursor(e.target.selectionStart ?? e.target.value.length);
          requestAnimationFrame(syncTextareaHeight);
        }}
        onPaste={handlePaste}
        onClick={syncCursor}
        onKeyUp={() => {
          syncCursor();
          syncTextareaHeight();
        }}
        onSelect={syncCursor}
        placeholder={placeholder}
        rows={1}
        disabled={inputDisabled}
        onCompositionStart={() => {
          composingRef.current = true;
        }}
        onCompositionEnd={() => {
          composingRef.current = false;
          syncTextareaHeight();
        }}
        onKeyDown={(e) => {
          if (showMenu) {
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setActiveIndex((i) => (i + 1) % suggestions.length);
              return;
            }
            if (e.key === 'ArrowUp') {
              e.preventDefault();
              setActiveIndex((i) => (i - 1 + suggestions.length) % suggestions.length);
              return;
            }
            if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
              e.preventDefault();
              const item = suggestions[activeIndex];
              if (item) {
                applySuggestion(item);
              }
              return;
            }
            if (e.key === 'Escape') {
              e.preventDefault();
              setMenuOpen(false);
              return;
            }
          }

          if (e.key !== 'Enter' || e.shiftKey) {
            return;
          }
          if (composingRef.current || e.nativeEvent.isComposing || e.keyCode === 229) {
            return;
          }
          e.preventDefault();
          if (!busy && canSend) {
            onSend();
          }
        }}
      />
      <div className="cursor-composer-bar">
        <span className="cursor-composer-hint">{hintText}</span>
        <div className="cursor-composer-bar-actions">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT_IMAGE_TYPES}
            multiple
            hidden
            onChange={(e) => {
              void addFiles(e.target.files);
              e.target.value = '';
            }}
          />
          <button
            type="button"
            className="cursor-composer-attach"
            disabled={inputDisabled}
            title={IMAGE_ATTACH_TOOLTIP}
            aria-label={IMAGE_ATTACH_TOOLTIP}
            onClick={() => fileInputRef.current?.click()}
          >
            <svg viewBox="0 0 16 16" aria-hidden="true" className="cursor-composer-attach-icon">
              <path
                d="M8.5 2.5a3.5 3.5 0 0 0-4.95 4.95l4.6 4.6a2.5 2.5 0 0 0 3.54-3.54L7.2 5.2a1.5 1.5 0 0 0-2.12 2.12l4.25 4.25"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          {busy && onStop && !stopInStatusBar ? (
            <button
              type="button"
              className="cursor-composer-stop"
              onClick={onStop}
              aria-label="停止"
            >
              停止
            </button>
          ) : (
            <button
              type="button"
              className="cursor-composer-send"
              disabled={!canSend || busy}
              onClick={onSend}
              aria-label="发送"
            >
              <svg viewBox="0 0 16 16" aria-hidden="true" className="cursor-composer-send-icon">
                <path
                  d="M8 12V4M8 4L5 7M8 4l3 3"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
    </>
  );

  const body = (
    <div
      className={`cursor-composer-body${dragOver ? ' is-drag-over' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {inner}
    </div>
  );

  if (embedded) {
    return body;
  }

  return (
    <div className="cursor-composer-wrap">
      <div className="cursor-composer">{body}</div>
    </div>
  );
}
