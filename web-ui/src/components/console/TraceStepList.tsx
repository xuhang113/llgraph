import { useEffect, useState } from 'react';
import type { TraceStep } from '../../types/trace';
import {
  filterTraceStepsForDisplay,
  formatInvokeTiming,
  formatStepElapsedLabel,
  formatStepUsage,
  parseTraceTurnsFromRemote,
  stepMarker,
} from '../../types/trace';
import { parseTraceStep } from '../../pages/console/traceUtils';
import MarkdownView from './MarkdownView';
import { api } from '../../api/client';

interface Props {
  steps: TraceStep[];
  /** 默认展开最后一步（进行中 trace） */
  defaultOpenLast?: boolean;
  /** 折叠态 body 预览行数 */
  previewLines?: number;
  /** all 模式：展示完整 body，不截断 */
  expandBodies?: boolean;
  /** 拉 explore 子 Trace 用 */
  slug?: string;
}

const DEFAULT_PREVIEW = 12;

function resolveExploreSubThread(step: TraceStep): string {
  const direct = (step.sub_thread || '').trim();
  if (direct) {
    return direct;
  }
  for (const line of step.body_lines ?? []) {
    const m = /^sub_thread=(.+)$/.exec(line.trim());
    if (m?.[1]?.trim()) {
      return m[1].trim();
    }
  }
  return '';
}

function ExploreChildTrace({
  slug,
  subThread,
  enabled,
}: {
  slug: string;
  subThread: string;
  enabled: boolean;
}) {
  const [childSteps, setChildSteps] = useState<TraceStep[] | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError('');
    void api
      .sessionTrace(slug, subThread)
      .then((remote) => {
        if (cancelled) {
          return;
        }
        const fromTurns = parseTraceTurnsFromRemote(remote.turns, []).flatMap((t) => t.steps);
        const fromSteps = Array.isArray(remote.steps)
          ? remote.steps
              .filter((s): s is Record<string, unknown> => s != null && typeof s === 'object')
              .map((s) => parseTraceStep(s))
          : [];
        const merged = fromTurns.length > 0 ? fromTurns : fromSteps;
        setChildSteps(filterTraceStepsForDisplay(merged));
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setChildSteps([]);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [slug, subThread, enabled]);

  if (!enabled) {
    return null;
  }
  if (loading && !childSteps) {
    return <div className="cursor-trace-step-line">加载子 Agent Trace…</div>;
  }
  if (error) {
    return <div className="cursor-trace-step-line">加载失败：{error}</div>;
  }
  if (!childSteps || childSteps.length === 0) {
    return <div className="cursor-trace-step-line">子会话暂无步骤（可在侧栏点进查看）</div>;
  }
  return (
    <div className="cursor-trace-explore-child">
      <div className="cursor-trace-steps-label">子 Agent · {subThread}</div>
      <TraceStepList steps={childSteps} defaultOpenLast={false} previewLines={8} />
    </div>
  );
}

function TraceStepItem({
  step,
  index,
  total,
  defaultOpenLast,
  previewLines,
  expandBodies,
  slug,
}: {
  step: TraceStep;
  index: number;
  total: number;
  defaultOpenLast: boolean;
  previewLines: number;
  expandBodies: boolean;
  slug?: string;
}) {
  const openByDefault = defaultOpenLast && index === total - 1;
  const [open, setOpen] = useState(openByDefault);
  const body = step.body_lines ?? [];
  // Think / reply：展开即看全文；其它 kind 在 steps 模式仍可预览截断
  const showFullBody =
    expandBodies || step.kind === 'thinking' || step.kind === 'reply';
  const lineLimit = showFullBody ? body.length : previewLines;
  const marker = stepMarker(step);
  const usageText = formatStepUsage(step.usage);
  const timingText = formatInvokeTiming(step.invoke_timing);
  const summaryText = [step.summary, usageText, timingText].filter(Boolean).join('  ');
  const exploreThread = step.kind === 'explore' ? resolveExploreSubThread(step) : '';
  const previewBody = body.filter((line) => !line.trim().startsWith('sub_thread='));

  return (
    <details
      className={`cursor-trace-step${
        step.kind === 'search_params' ? ' cursor-trace-step--search-params' : ''
      }${step.kind === 'explore' ? ' cursor-trace-step--explore' : ''}${
        step.kind === 'thinking' ? ' cursor-trace-step--thinking' : ''
      }`}
      open={open}
      onToggle={(event) => {
        setOpen((event.currentTarget as HTMLDetailsElement).open);
      }}
    >
      <summary className="cursor-trace-step-summary">
        <span className="cursor-trace-step-marker">{marker}</span>
        <span className="cursor-trace-step-title">
          #{index + 1} {step.title}
        </span>
        <span className="cursor-trace-step-meta">({formatStepElapsedLabel(step)})</span>
        {summaryText && <span className="cursor-trace-step-summary-text">{summaryText}</span>}
      </summary>
      <div className="cursor-trace-step-body">
        {step.kind === 'reply' ? (
          <MarkdownView
            content={previewBody.slice(0, lineLimit).join('\n')}
            streaming={defaultOpenLast && index === total - 1}
          />
        ) : (
          previewBody.slice(0, lineLimit).map((line, lineIndex) => (
            <div
              key={lineIndex}
              className={
                line.trimStart().startsWith('【规划】')
                  ? 'cursor-trace-step-line cursor-trace-step-line--plan'
                  : step.kind === 'search_params'
                    ? 'cursor-trace-step-line cursor-trace-step-line--search-params'
                    : step.kind === 'thinking'
                      ? 'cursor-trace-step-line cursor-trace-step-line--thinking'
                      : 'cursor-trace-step-line'
              }
            >
              {line}
            </div>
          ))
        )}
        {!showFullBody && previewBody.length > previewLines && (
          <div className="cursor-trace-step-more">… 还有 {previewBody.length - previewLines} 行</div>
        )}
        {exploreThread && slug ? (
          <ExploreChildTrace slug={slug} subThread={exploreThread} enabled={open} />
        ) : null}
        {exploreThread && !slug ? (
          <div className="cursor-trace-step-line">子会话：{exploreThread}</div>
        ) : null}
      </div>
    </details>
  );
}

export default function TraceStepList({
  steps,
  defaultOpenLast = true,
  previewLines = DEFAULT_PREVIEW,
  expandBodies = false,
  slug,
}: Props) {
  const visible = filterTraceStepsForDisplay(steps);
  if (visible.length === 0) {
    return null;
  }

  return (
    <div className="cursor-trace-step-list">
      {visible.map((step, index) => (
        <TraceStepItem
          key={`${step.step_id}-${index}`}
          step={step}
          index={index}
          total={visible.length}
          defaultOpenLast={defaultOpenLast}
          previewLines={previewLines}
          expandBodies={expandBodies}
          slug={slug}
        />
      ))}
    </div>
  );
}
