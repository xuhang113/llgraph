import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

interface Props {
  svgHtml: string;
  onClose: () => void;
}

type ViewBox = { x: number; y: number; width: number; height: number };

const MIN_ZOOM = 0.85;
const MAX_ZOOM = 3.5;
const WHEEL_ZOOM_SENSITIVITY = 0.00085;
const PAN_SENSITIVITY = 0.86;
const BUTTON_ZOOM_IN = 1.125;
const BUTTON_ZOOM_OUT = 1 / BUTTON_ZOOM_IN;
const SCROLL_LOCK_SELECTOR = '.cursor-main-scroll';

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function wheelZoomFactor(delta: number): number {
  const sign = delta > 0 ? 1 : -1;
  const oldFactor = sign > 0 ? 0.92 : 1.08;
  const damped = clamp(Math.abs(delta), 0, 120) * sign;
  const newFactor = Math.exp(-damped * WHEEL_ZOOM_SENSITIVITY);
  return 1 + 0.5 * (oldFactor - 1 + (newFactor - 1));
}

function measureDiagramBounds(svg: SVGSVGElement): ViewBox {
  const vb = svg.viewBox?.baseVal;
  let box: ViewBox;
  try {
    const b = svg.getBBox();
    if (b.width > 0 && b.height > 0) {
      box = { x: b.x, y: b.y, width: b.width, height: b.height };
    } else if (vb && vb.width > 0 && vb.height > 0) {
      box = { x: vb.x, y: vb.y, width: vb.width, height: vb.height };
    } else {
      box = { x: 0, y: 0, width: 1, height: 1 };
    }
  } catch {
    box =
      vb && vb.width > 0 && vb.height > 0
        ? { x: vb.x, y: vb.y, width: vb.width, height: vb.height }
        : { x: 0, y: 0, width: 1, height: 1 };
  }
  const pad = Math.max(12, Math.max(box.width, box.height) * 0.06);
  return {
    x: box.x - pad,
    y: box.y - pad,
    width: box.width + pad * 2,
    height: box.height + pad * 2,
  };
}

function computeContainViewBox(
  content: ViewBox,
  stagePxW: number,
  stagePxH: number,
  paddingPx: number,
): ViewBox {
  const innerW = Math.max(stagePxW - paddingPx * 2, 1);
  const innerH = Math.max(stagePxH - paddingPx * 2, 1);
  const stageAspect = innerW / innerH;
  const contentAspect = content.width / content.height;

  let vbW = content.width;
  let vbH = content.height;
  if (contentAspect > stageAspect) {
    vbH = content.width / stageAspect;
  } else {
    vbW = content.height * stageAspect;
  }
  const cx = content.x + content.width / 2;
  const cy = content.y + content.height / 2;
  return {
    x: cx - vbW / 2,
    y: cy - vbH / 2,
    width: vbW,
    height: vbH,
  };
}

function applyViewBox(svg: SVGSVGElement, viewBox: ViewBox) {
  svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.style.width = '100%';
  svg.style.height = '100%';
  svg.style.maxWidth = 'none';
  svg.style.display = 'block';
}

function zoomPercent(fitBase: ViewBox, current: ViewBox): number {
  return Math.round((fitBase.width / current.width) * 100);
}

function lockPageScroll(): () => void {
  const lockedClass = 'cursor-scroll-locked';
  const bodyOverflow = document.body.style.overflow;
  document.body.style.overflow = 'hidden';
  document.documentElement.classList.add(lockedClass);
  const scrollers = Array.from(
    document.querySelectorAll<HTMLElement>(SCROLL_LOCK_SELECTOR),
  );
  scrollers.forEach((el) => el.classList.add(lockedClass));
  return () => {
    document.body.style.overflow = bodyOverflow;
    document.documentElement.classList.remove(lockedClass);
    scrollers.forEach((el) => el.classList.remove(lockedClass));
  };
}

export default function MermaidLightbox({ svgHtml, onClose }: Props) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<ViewBox | null>(null);
  const fitBaseRef = useRef<ViewBox | null>(null);
  const [viewBox, setViewBox] = useState<ViewBox | null>(null);
  const dragging = useRef(false);
  const lastPointer = useRef({ x: 0, y: 0 });

  const fitToStage = useCallback(() => {
    const content = contentRef.current;
    const stage = stageRef.current;
    if (!content || !stage) {
      return;
    }
    const rect = stage.getBoundingClientRect();
    const fitted = computeContainViewBox(content, rect.width, rect.height, 28);
    fitBaseRef.current = fitted;
    setViewBox({ ...fitted });
  }, []);

  const resetZoom = useCallback(() => {
    const fitted = fitBaseRef.current;
    if (fitted) {
      setViewBox({ ...fitted });
    } else {
      fitToStage();
    }
  }, [fitToStage]);

  useEffect(() => {
    const unlock = lockPageScroll();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    const blockBackgroundScroll = (e: Event) => {
      const overlay = overlayRef.current;
      if (!overlay) {
        return;
      }
      const target = e.target;
      if (target instanceof Node && overlay.contains(target)) {
        e.preventDefault();
      }
    };
    window.addEventListener('keydown', onKey);
    document.addEventListener('wheel', blockBackgroundScroll, { passive: false, capture: true });
    document.addEventListener('touchmove', blockBackgroundScroll, { passive: false, capture: true });
    return () => {
      unlock();
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('wheel', blockBackgroundScroll, { capture: true });
      document.removeEventListener('touchmove', blockBackgroundScroll, { capture: true });
    };
  }, [onClose]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    canvas.innerHTML = svgHtml;
    const svg = canvas.querySelector('svg');
    if (!svg) {
      return;
    }

    const init = () => {
      const content = measureDiagramBounds(svg);
      contentRef.current = content;
      fitToStage();
    };

    requestAnimationFrame(() => {
      requestAnimationFrame(init);
    });
  }, [svgHtml, fitToStage]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !contentRef.current) {
      return undefined;
    }
    const observer = new ResizeObserver(() => {
      if (!fitBaseRef.current) {
        fitToStage();
      }
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, [svgHtml, fitToStage]);

  useEffect(() => {
    const svg = canvasRef.current?.querySelector('svg');
    if (!svg || !viewBox) {
      return;
    }
    applyViewBox(svg, viewBox);
  }, [viewBox, svgHtml]);

  const zoomAt = useCallback((factor: number, clientX?: number, clientY?: number) => {
    const fitBase = fitBaseRef.current;
    const content = contentRef.current;
    const stage = stageRef.current;
    if (!fitBase || !content || !stage) {
      return;
    }
    const rect = stage.getBoundingClientRect();
    const px = clientX !== undefined ? (clientX - rect.left) / rect.width : 0.5;
    const py = clientY !== undefined ? (clientY - rect.top) / rect.height : 0.5;

    setViewBox((prev) => {
      if (!prev) {
        return prev;
      }
      const currentZoom = fitBase.width / prev.width;
      const nextZoom = clamp(currentZoom * factor, MIN_ZOOM, MAX_ZOOM);
      const nextWidth = fitBase.width / nextZoom;
      const nextHeight = fitBase.height / nextZoom;
      const focusX = prev.x + px * prev.width;
      const focusY = prev.y + py * prev.height;
      return {
        x: focusX - px * nextWidth,
        y: focusY - py * nextHeight,
        width: nextWidth,
        height: nextHeight,
      };
    });
  }, []);

  const onWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    let delta = e.deltaY;
    if (e.deltaMode === 1) {
      delta *= 16;
    } else if (e.deltaMode === 2) {
      delta *= window.innerHeight;
    }
    zoomAt(wheelZoomFactor(delta), e.clientX, e.clientY);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) {
      return;
    }
    dragging.current = true;
    lastPointer.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();
    e.stopPropagation();
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current || !viewBox) {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    const stage = stageRef.current;
    if (!stage) {
      return;
    }
    const rect = stage.getBoundingClientRect();
    const dx = (e.clientX - lastPointer.current.x) * PAN_SENSITIVITY;
    const dy = (e.clientY - lastPointer.current.y) * PAN_SENSITIVITY;
    lastPointer.current = { x: e.clientX, y: e.clientY };
    const dxView = (-dx / rect.width) * viewBox.width;
    const dyView = (-dy / rect.height) * viewBox.height;
    setViewBox((prev) =>
      prev
        ? {
            ...prev,
            x: prev.x + dxView,
            y: prev.y + dyView,
          }
        : prev,
    );
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) {
      return;
    }
    dragging.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    e.preventDefault();
    e.stopPropagation();
  };

  const fitBase = fitBaseRef.current;
  const displayZoom = fitBase && viewBox ? zoomPercent(fitBase, viewBox) : 100;

  const panel = (
    <div
      ref={overlayRef}
      className="cursor-mermaid-lightbox"
      role="dialog"
      aria-modal="true"
      aria-label="图表预览"
      onClick={onClose}
    >
      <div className="cursor-mermaid-lightbox-panel" onClick={(e) => e.stopPropagation()}>
        <header className="cursor-mermaid-lightbox-toolbar">
          <span className="cursor-mermaid-lightbox-title">图表预览</span>
          <span className="cursor-mermaid-lightbox-zoom">{displayZoom}%</span>
          <button type="button" aria-label="缩小" onClick={() => zoomAt(BUTTON_ZOOM_OUT)}>
            −
          </button>
          <button type="button" aria-label="放大" onClick={() => zoomAt(BUTTON_ZOOM_IN)}>
            +
          </button>
          <button type="button" onClick={fitToStage}>
            适应面板
          </button>
          <button type="button" onClick={resetZoom}>
            重置
          </button>
          <button
            type="button"
            className="cursor-mermaid-lightbox-close"
            aria-label="关闭预览"
            onClick={onClose}
          >
            ×
          </button>
        </header>
        <div
          ref={stageRef}
          className="cursor-mermaid-lightbox-stage"
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <div ref={canvasRef} className="cursor-mermaid-lightbox-canvas" />
        </div>
      </div>
    </div>
  );

  return createPortal(panel, document.body);
}
