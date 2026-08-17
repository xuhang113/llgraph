import { useCallback, useEffect, useRef, useState, type RefCallback } from 'react';

const BOTTOM_THRESHOLD_PX = 64;

function isAtBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD_PX;
}

/**
 * 日志式滚动：内容更新时贴底；用户上滑/拖滚动条后暂停，滑回底部再恢复。
 *
 * 注意：内容增高时 scrollTop 不变会导致暂时「不在底部」，此时仍应跟滚（pinned），
 * 不能把「未在底部」当成用户主动上滑。
 */
export function useStickToBottomScroll<T extends HTMLElement>(
  contentDeps: readonly unknown[],
  options?: { enabled?: boolean; resetKey?: string | number; forcePin?: boolean },
): { ref: RefCallback<T>; stickToBottom: () => void; pinned: boolean } {
  const elRef = useRef<T | null>(null);
  const [scrollRoot, setScrollRoot] = useState<T | null>(null);
  const pinnedRef = useRef(true);
  const [pinned, setPinned] = useState(true);
  const programmaticScrollRef = useRef(false);
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
    programmaticScrollRef.current = true;
    el.scrollTop = el.scrollHeight;
    // 下一帧再清标记，避免 scroll 事件把 pinned 误清掉
    requestAnimationFrame(() => {
      programmaticScrollRef.current = false;
      // 布局未稳时再贴一次
      if (elRef.current && (pinnedRef.current || forcePin)) {
        elRef.current.scrollTop = elRef.current.scrollHeight;
      }
    });
  }, [forcePin]);

  const stickToBottom = useCallback(() => {
    syncPinned(true);
    scrollToBottom();
  }, [scrollToBottom, syncPinned]);

  /** 是否跟滚：只看 pinned / forcePin，不要求此刻已在底部。 */
  const shouldAutoStick = useCallback((): boolean => {
    if (forcePin) {
      return true;
    }
    return pinnedRef.current;
  }, [forcePin]);

  // 切换会话：重置为贴底，并多拍几次等历史/Markdown 布局
  useEffect(() => {
    syncPinned(true);
    scrollToBottom();
    const t0 = window.setTimeout(scrollToBottom, 0);
    const t1 = window.setTimeout(scrollToBottom, 120);
    const t2 = window.setTimeout(scrollToBottom, 320);
    return () => {
      window.clearTimeout(t0);
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [resetKey, scrollToBottom, syncPinned]);

  useEffect(() => {
    const el = scrollRoot;
    if (!el || !enabled) {
      return;
    }
    const onScroll = () => {
      if (programmaticScrollRef.current) {
        return;
      }
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
    scrollToBottom();
  }, [enabled, forcePin, scrollToBottom, shouldAutoStick, syncPinned, ...contentDeps]);

  /** DOM 子树增高（trace 展开、流式、Markdown）时贴底 */
  useEffect(() => {
    const el = scrollRoot;
    if (!el || !enabled) {
      return;
    }
    let raf = 0;
    const scheduleStick = () => {
      if (!shouldAutoStick()) {
        return;
      }
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
