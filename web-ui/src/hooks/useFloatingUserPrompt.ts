import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from 'react';
import type { ChatMessage } from '../components/console/ChatThread';
import type { ChatImageAttachment } from '../types/chatImage';

export interface FloatingUserState {
  /** 浮动栏是否展示（含退出动画期间） */
  visible: boolean;
  messageId: string | null;
  text: string;
  images?: ChatImageAttachment[];
}

const FLOAT_TOP_INSET_PX = 16;
/** 阅读锚点：scrollTop + 该偏移以下的最近 user 为当前轮 */
const READ_ANCHOR_PX = 96;
/** 内联用户气泡仍完整露在顶部时隐藏浮动栏 */
const INLINE_BOTTOM_MIN_PX = 36;
const HIDE_DELAY_MS = 150;

function offsetTopInScrollRoot(el: HTMLElement, root: HTMLElement): number {
  const elRect = el.getBoundingClientRect();
  const rootRect = root.getBoundingClientRect();
  return elRect.top - rootRect.top + root.scrollTop;
}

function isInlineUserVisibleAtTop(el: HTMLElement, stickyTop: number): boolean {
  const rect = el.getBoundingClientRect();
  return rect.top >= stickyTop - 6 && rect.bottom > stickyTop + INLINE_BOTTOM_MIN_PX;
}

export function useFloatingUserPrompt(
  scrollRootRef: RefObject<HTMLElement | null> | undefined,
  userElementRefs: RefObject<Map<string, HTMLElement>>,
  messages: ChatMessage[],
): FloatingUserState & { scrollToUser: (id: string) => void } {
  const userMessages = useMemo(
    () =>
      messages.filter(
        (m) => m.role === 'user' && (m.text.trim().length > 0 || (m.images?.length ?? 0) > 0),
      ),
    [messages],
  );

  const [state, setState] = useState<FloatingUserState>({
    visible: false,
    messageId: null,
    text: '',
  });

  const hideTimerRef = useRef<number | null>(null);

  const clearHideTimer = useCallback(() => {
    if (hideTimerRef.current != null) {
      window.clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
  }, []);

  const applyVisible = useCallback((next: FloatingUserState) => {
    setState((prev) => {
      if (
        prev.visible === next.visible &&
        prev.messageId === next.messageId &&
        prev.text === next.text &&
        prev.images === next.images
      ) {
        return prev;
      }
      return next;
    });
  }, []);

  const beginHide = useCallback(() => {
    clearHideTimer();
    setState((prev) => {
      if (!prev.messageId) {
        return { ...prev, visible: false };
      }
      hideTimerRef.current = window.setTimeout(() => {
        hideTimerRef.current = null;
        applyVisible({ visible: false, messageId: null, text: '' });
      }, HIDE_DELAY_MS);
      if (!prev.visible) {
        return prev;
      }
      return { ...prev, visible: false };
    });
  }, [applyVisible, clearHideTimer]);

  const update = useCallback(() => {
    const root = scrollRootRef?.current;
    const refs = userElementRefs.current;
    if (!root || !refs || userMessages.length === 0) {
      clearHideTimer();
      applyVisible({ visible: false, messageId: null, text: '' });
      return;
    }

    const rootRect = root.getBoundingClientRect();
    const stickyTop = rootRect.top + FLOAT_TOP_INSET_PX;
    const readAnchor = root.scrollTop + READ_ANCHOR_PX;

    let activeUser: ChatMessage | null = null;
    let activeEl: HTMLElement | null = null;

    for (const msg of userMessages) {
      const el = refs.get(msg.id);
      if (!el) {
        continue;
      }
      if (offsetTopInScrollRoot(el, root) <= readAnchor) {
        activeUser = msg;
        activeEl = el;
      }
    }

    if (!activeUser || !activeEl) {
      beginHide();
      return;
    }

    const shouldShow = !isInlineUserVisibleAtTop(activeEl, stickyTop);
    if (!shouldShow) {
      beginHide();
      return;
    }

    clearHideTimer();

    applyVisible({
      visible: true,
      messageId: activeUser.id,
      text: activeUser.text,
      images: activeUser.images,
    });
  }, [
    scrollRootRef,
    userElementRefs,
    userMessages,
    applyVisible,
    clearHideTimer,
    beginHide,
  ]);

  useEffect(() => {
    const root = scrollRootRef?.current;
    if (!root) {
      return;
    }
    update();
    root.addEventListener('scroll', update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(root);
    return () => {
      root.removeEventListener('scroll', update);
      ro.disconnect();
      clearHideTimer();
    };
  }, [scrollRootRef, update, userMessages.length, messages.length, clearHideTimer]);

  const scrollToUser = useCallback(
    (id: string) => {
      const el = userElementRefs.current?.get(id);
      const root = scrollRootRef?.current;
      if (!el || !root) {
        return;
      }
      const rootRect = root.getBoundingClientRect();
      const elRect = el.getBoundingClientRect();
      const target = root.scrollTop + (elRect.top - rootRect.top) - 64;
      root.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
    },
    [scrollRootRef, userElementRefs],
  );

  return { ...state, scrollToUser };
}
