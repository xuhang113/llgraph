import { useEffect, useMemo, useRef, useState } from 'react';
import type { SlashCatalogItem } from '../../api/client';
import { filterSlashCatalog, parseSlashPartial } from '../../utils/slashCatalog';
import SlashCompletionMenu from './SlashCompletionMenu';

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop?: () => void;
  busy?: boolean;
  disabled?: boolean;
  placeholder: string;
  slashCatalog?: SlashCatalogItem[];
  /** 已包在 ComposerDock 内，不重复外层 wrap */
  embedded?: boolean;
  /** Stop 按钮在顶栏状态条，底部不重复 */
  stopInStatusBar?: boolean;
}

const COMPOSER_MIN_HEIGHT = 44;
const COMPOSER_MAX_HEIGHT = 240;

export default function ChatComposer({
  value,
  onChange,
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
  const [cursor, setCursor] = useState(0);
  const [menuOpen, setMenuOpen] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);

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

  const hintText = showMenu
    ? '↑↓ 选择 · Tab/Enter 补全 · Esc 关闭'
    : busy
      ? '生成中…'
      : 'Enter 发送 · Shift+Enter 换行 · / 命令补全';

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
            if (!busy) {
              onSend();
            }
          }}
        />
        <div className="cursor-composer-bar">
          <span className="cursor-composer-hint">{hintText}</span>
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
              disabled={disabled || !value.trim()}
              onClick={onSend}
              aria-label="发送"
            >
              ↑
            </button>
          )}
        </div>
    </>
  );

  if (embedded) {
    return <div className="cursor-composer-body">{inner}</div>;
  }

  return (
    <div className="cursor-composer-wrap">
      <div className="cursor-composer">{inner}</div>
    </div>
  );
}
