import { useCallback, useEffect, useRef, type RefObject } from 'react';

const BOTTOM_THRESHOLD_PX = 48;

function isAtBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD_PX;
}

/**
 * 日志式滚动：内容更新时贴底；用户上滑后暂停，滑回底部再恢复。
 */
export function useStickToBottomScroll<T extends HTMLElement>(
  contentDeps: readonly unknown[],
  options?: { enabled?: boolean; resetKey?: string | number; forcePin?: boolean },
): { ref: RefObject<T | null>; stickToBottom: () => void } {
  const ref = useRef<T | null>(null);
  const pinnedRef = useRef(true);
  const enabled = options?.enabled !== false;
  const forcePin = options?.forcePin === true;
  const resetKey = options?.resetKey;

  const scrollToBottom = useCallback(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    el.scrollTop = el.scrollHeight;
  }, []);

  const stickToBottom = useCallback(() => {
    pinnedRef.current = true;
    scrollToBottom();
  }, [scrollToBottom]);

  useEffect(() => {
    pinnedRef.current = true;
    requestAnimationFrame(() => scrollToBottom());
  }, [resetKey, scrollToBottom]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !enabled) {
      return;
    }
    const onScroll = () => {
      pinnedRef.current = isAtBottom(el);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    if (forcePin) {
      pinnedRef.current = true;
    }
    if (!forcePin && !pinnedRef.current) {
      return;
    }
    const el = ref.current;
    if (!el) {
      return;
    }
    const scroll = () => {
      if (forcePin || pinnedRef.current) {
        scrollToBottom();
      }
    };
    requestAnimationFrame(() => requestAnimationFrame(scroll));
  }, [enabled, forcePin, scrollToBottom, ...contentDeps]);

  /** DOM 子树增高（trace 步骤展开、流式追加）时贴底 */
  useEffect(() => {
    const el = ref.current;
    if (!el || !enabled) {
      return;
    }
    let raf = 0;
    const scheduleStick = () => {
      if (raf) {
        cancelAnimationFrame(raf);
      }
      raf = requestAnimationFrame(() => {
        if (forcePin || pinnedRef.current) {
          scrollToBottom();
        }
      });
    };
    const observer = new MutationObserver(scheduleStick);
    observer.observe(el, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    return () => {
      observer.disconnect();
      if (raf) {
        cancelAnimationFrame(raf);
      }
    };
  }, [enabled, forcePin, scrollToBottom, resetKey]);

  return { ref, stickToBottom };
}
