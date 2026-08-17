import { useEffect, useId, useRef, useState } from 'react';
import { buildMermaidConfig } from './mermaidTheme';
import MermaidLightbox from './MermaidLightbox';
import CopyButton from './CopyButton';
import { toMermaidMarkdown } from '../../utils/clipboard';

interface Props {
  code: string;
}

function removeMermaidRenderHostArtifacts(renderId: string) {
  for (const id of [`d${renderId}`, `i${renderId}`]) {
    document.getElementById(id)?.remove();
  }
}

export default function MermaidChart({ code }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const reactId = useId().replace(/:/g, '');
  const [error, setError] = useState<string | null>(null);
  const [svgHtml, setSvgHtml] = useState<string | null>(null);
  const [lightboxOpen, setLightboxOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const source = code.trim();
    if (!source) {
      return undefined;
    }

    void (async () => {
      const renderId = `mermaid-${reactId}-${Date.now()}`;
      const host = document.createElement('div');
      host.setAttribute('aria-hidden', 'true');
      host.style.cssText =
        'position:fixed;left:-10000px;top:0;width:1200px;height:1200px;overflow:hidden;visibility:hidden;pointer-events:none';
      document.body.appendChild(host);

      try {
        const mermaid = await import('mermaid');
        mermaid.default.initialize(buildMermaidConfig());
        if (cancelled) {
          return;
        }
        const { svg } = await mermaid.default.render(renderId, source, host);
        if (!cancelled) {
          setSvgHtml(svg);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setSvgHtml(null);
          setError(err instanceof Error ? err.message : 'Mermaid 渲染失败');
        }
      } finally {
        host.remove();
        removeMermaidRenderHostArtifacts(renderId);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [code, reactId]);

  useEffect(() => {
    if (containerRef.current && svgHtml) {
      containerRef.current.innerHTML = svgHtml;
    }
  }, [svgHtml]);

  if (error) {
    return (
      <div className="markdown-mermaid markdown-mermaid--error">
        <CopyButton
          text={toMermaidMarkdown(code)}
          title="复制 Mermaid"
          className="markdown-mermaid-copy"
          iconOnly
        />
        <div className="markdown-mermaid-error-title">Mermaid 图表无法渲染</div>
        <pre>
          <code>{code}</code>
        </pre>
      </div>
    );
  }

  return (
    <>
      <div className="markdown-mermaid-wrap">
        <CopyButton
          text={toMermaidMarkdown(code)}
          title="复制 Mermaid"
          className="markdown-mermaid-copy"
          iconOnly
        />
        <button
          type="button"
          className={`markdown-mermaid markdown-mermaid--interactive${svgHtml ? ' is-ready' : ''}`}
          aria-label="预览图表"
          disabled={!svgHtml}
          onClick={() => {
            if (svgHtml) {
              setLightboxOpen(true);
            }
          }}
        >
          <div ref={containerRef} className="markdown-mermaid-inline" />
        </button>
      </div>
      {lightboxOpen && svgHtml && (
        <MermaidLightbox svgHtml={svgHtml} onClose={() => setLightboxOpen(false)} />
      )}
    </>
  );
}
