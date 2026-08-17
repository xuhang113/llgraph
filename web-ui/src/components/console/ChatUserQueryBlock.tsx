import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from 'react';
import CopyButton from './CopyButton';
import ChatImageStrip from './ChatImageStrip';
import type { ChatImageAttachment } from '../../types/chatImage';

interface Props {
  text: string;
  images?: ChatImageAttachment[];
  /** 折叠为固定行数（浮动栏 / 超长气泡） */
  collapsed?: boolean;
  /** 允许展开/收起 */
  expandable?: boolean;
  /**
   * 展开后正文最大高度。浮动条宜小，避免盖住聊天与输入框；
   * 内联超长气泡也需限高，保证主滚动容器可滚动。
   */
  expandedMaxHeight?: string;
}

export default function ChatUserQueryBlock({
  text,
  images,
  collapsed = false,
  expandable = false,
  expandedMaxHeight = 'min(42vh, 360px)',
}: Props) {
  const trimmed = text.trim();
  const [expanded, setExpanded] = useState(false);
  const [isTruncated, setIsTruncated] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setExpanded(false);
  }, [text]);

  const measureTruncation = useCallback(() => {
    const el = textRef.current;
    if (!el || !collapsed || expanded) {
      setIsTruncated(false);
      return;
    }
    setIsTruncated(el.scrollHeight > el.clientHeight + 1);
  }, [collapsed, expanded]);

  useLayoutEffect(() => {
    measureTruncation();
  }, [trimmed, collapsed, expanded, measureTruncation]);

  useEffect(() => {
    const el = textRef.current;
    if (!el) {
      return;
    }
    const ro = new ResizeObserver(measureTruncation);
    ro.observe(el);
    return () => ro.disconnect();
  }, [measureTruncation]);

  useEffect(() => {
    if (!expandable || !expanded) {
      return;
    }
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        setExpanded(false);
      }
    };
    const onPointerDown = (event: PointerEvent) => {
      const root = rootRef.current;
      if (!root || !(event.target instanceof Node)) {
        return;
      }
      if (!root.contains(event.target)) {
        setExpanded(false);
      }
    };
    window.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointerDown, true);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointerDown, true);
    };
  }, [expandable, expanded]);

  if (!trimmed && (images?.length ?? 0) === 0) {
    return null;
  }

  const isCollapsed = collapsed && !expanded;
  const canExpand = expandable && collapsed && isTruncated && !expanded;
  const canCollapse = expandable && expanded;

  const toggleExpanded = () => {
    if (canExpand) {
      setExpanded(true);
      return;
    }
    if (canCollapse) {
      setExpanded(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent) => {
    if (!canExpand && !canCollapse) {
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleExpanded();
    }
  };

  return (
    <>
      {(images?.length ?? 0) > 0 && <ChatImageStrip images={images ?? []} />}
      {trimmed && (
        <div
          ref={rootRef}
          className={`cursor-user-query${isCollapsed ? ' cursor-user-query--collapsed' : ''}${
            expandable ? ' cursor-user-query--expandable' : ''
          }${expanded ? ' is-expanded' : ''}${isTruncated && isCollapsed ? ' is-truncated' : ''}`}
          style={
            expanded
              ? ({
                  '--cursor-user-query-expanded-max': expandedMaxHeight,
                } as CSSProperties)
              : undefined
          }
          onClick={canExpand ? toggleExpanded : undefined}
          onKeyDown={canExpand || canCollapse ? handleKeyDown : undefined}
          role={canExpand || canCollapse ? 'button' : undefined}
          tabIndex={canExpand || canCollapse ? 0 : undefined}
          aria-expanded={canExpand || canCollapse ? expanded : undefined}
          title={canExpand ? '展开全文' : canCollapse ? '收起' : undefined}
        >
          <CopyButton
            text={trimmed}
            title="复制问题"
            className="cursor-user-query-copy"
            iconOnly
          />
          <div ref={textRef} className="cursor-user-query-text">
            {trimmed}
          </div>
          {canExpand && (
            <span className="cursor-user-query-expand-icon" aria-hidden="true">
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path
                  d="M10.5 5.5L5.5 10.5M10.5 5.5H7M10.5 5.5V9"
                  stroke="currentColor"
                  strokeWidth="1.35"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
          )}
          {canCollapse && (
            <button
              type="button"
              className="cursor-user-query-collapse"
              onClick={(event) => {
                event.stopPropagation();
                setExpanded(false);
              }}
              title="收起"
              aria-label="收起"
            >
              收起
            </button>
          )}
        </div>
      )}
    </>
  );
}
