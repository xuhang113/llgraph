import { useCallback, useEffect, useRef, useState, type RefCallback } from 'react';

const BOTTOM_THRESHOLD_PX = 64;

function isAtBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD_PX;
}

/**
 * 日志式滚动：内容更新时贴底；用户上滑/拖滚动条后暂停，滑回底部再恢复。
 */
export function useStickToBottomScroll<T extends HTMLElement>(
  contentDeps: readonly unknown[],
  options?: { enabled?: boolean; resetKey?: string | number; forcePin?: boolean },
): { ref: RefCallback<T>; stickToBottom: () => void; pinned: boolean } {
  const elRef = useRef<T | null>(null);
  const [scrollRoot, setScrollRoot] = useState<T | null>(null);
  const pinnedRef = useRef(true);
  const [pinned, setPinned] = useState(true);
  const enabled = options?.enabled !== false;
  const forcePin = options?.forcePin === true;
  const resetKey = options?.resetKey;

  const syncPinned = useCallback((next: boolean) => {
    pinnedRef.current = next;
    setPinned(next);
  }, []);

  const setRef = useCallback<RefCallback<T>>((node) => {
    elRef.current = node;
    setScrollRoot(node);
  }, []);

  const scrollToBottom = useCallback(() => {
    const el = elRef.current;
    if (!el) {
      return;
    }
    el.scrollTop = el.scrollHeight;
  }, []);

  const stickToBottom = useCallback(() => {
    syncPinned(true);
    scrollToBottom();
  }, [scrollToBottom, syncPinned]);

  const shouldAutoStick = useCallback((): boolean => {
    if (forcePin) {
      return true;
    }
    const el = elRef.current;
    if (!el) {
      return pinnedRef.current;
    }
    return pinnedRef.current && isAtBottom(el);
  }, [forcePin]);

  useEffect(() => {
    syncPinned(true);
    requestAnimationFrame(() => scrollToBottom());
  }, [resetKey, scrollToBottom, syncPinned]);

  useEffect(() => {
    const el = scrollRoot;
    if (!el || !enabled) {
      return;
    }
    const onScroll = () => {
      syncPinned(isAtBottom(el));
    };
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) {
        syncPinned(false);
        return;
      }
      if (event.deltaY > 0 && isAtBottom(el)) {
        syncPinned(true);
      }
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    el.addEventListener('wheel', onWheel, { passive: true });
    return () => {
      el.removeEventListener('scroll', onScroll);
      el.removeEventListener('wheel', onWheel);
    };
  }, [scrollRoot, enabled, syncPinned]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    if (forcePin) {
      syncPinned(true);
    }
    if (!shouldAutoStick()) {
      return;
    }
    const scroll = () => {
      if (shouldAutoStick()) {
        scrollToBottom();
      }
    };
    requestAnimationFrame(() => requestAnimationFrame(scroll));
  }, [enabled, forcePin, scrollToBottom, shouldAutoStick, syncPinned, ...contentDeps]);

  /** DOM 子树增高（trace 步骤展开、流式追加）时贴底 */
  useEffect(() => {
    const el = scrollRoot;
    if (!el || !enabled) {
      return;
    }
    let raf = 0;
    const scheduleStick = () => {
      if (raf) {
        cancelAnimationFrame(raf);
      }
      raf = requestAnimationFrame(() => {
        if (shouldAutoStick()) {
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
  }, [scrollRoot, enabled, shouldAutoStick, scrollToBottom, resetKey]);

  return { ref: setRef, stickToBottom, pinned };
}
