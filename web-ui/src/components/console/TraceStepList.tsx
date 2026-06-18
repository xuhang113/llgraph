import type { TraceStep } from '../../types/trace';
import {
  formatStepUsage,
  formatTraceDuration,
  stepMarker,
} from '../../types/trace';

interface Props {
  steps: TraceStep[];
  /** 默认展开最后一步（进行中 trace） */
  defaultOpenLast?: boolean;
  /** 折叠态 body 预览行数 */
  previewLines?: number;
  /** all 模式：展示完整 body，不截断 */
  expandBodies?: boolean;
}

const DEFAULT_PREVIEW = 12;

export default function TraceStepList({
  steps,
  defaultOpenLast = true,
  previewLines = DEFAULT_PREVIEW,
  expandBodies = false,
}: Props) {
  if (steps.length === 0) {
    return null;
  }

  return (
    <div className="cursor-trace-step-list">
      {steps.map((step, index) => {
        const body = step.body_lines ?? [];
        const open = defaultOpenLast && index === steps.length - 1;
        const lineLimit = expandBodies ? body.length : previewLines;
        const marker = stepMarker(step);
        const usageText = formatStepUsage(step.usage);
        const summaryText = [step.summary, usageText].filter(Boolean).join('  ');

        return (
          <details
            key={`${step.step_id}-${index}`}
            className={`cursor-trace-step${
              step.kind === 'search_params' ? ' cursor-trace-step--search-params' : ''
            }`}
            open={open}
          >
            <summary className="cursor-trace-step-summary">
              <span className="cursor-trace-step-marker">{marker}</span>
              <span className="cursor-trace-step-title">
                #{step.step_id} {step.title}
              </span>
              <span className="cursor-trace-step-meta">
                ({formatTraceDuration(step.elapsed)})
              </span>
              {summaryText && (
                <span className="cursor-trace-step-summary-text">{summaryText}</span>
              )}
            </summary>
            {body.length > 0 && (
              <div className="cursor-trace-step-body">
                {body.slice(0, lineLimit).map((line, lineIndex) => (
                  <div
                    key={lineIndex}
                    className={
                      line.trimStart().startsWith('【规划】')
                        ? 'cursor-trace-step-line cursor-trace-step-line--plan'
                        : step.kind === 'search_params'
                          ? 'cursor-trace-step-line cursor-trace-step-line--search-params'
                          : step.kind === 'thinking'
                            ? 'cursor-trace-step-line cursor-trace-step-line--thinking'
                            : step.kind === 'reply'
                              ? 'cursor-trace-step-line cursor-trace-step-line--reply'
                              : 'cursor-trace-step-line'
                    }
                  >
                    {line}
                  </div>
                ))}
                {!expandBodies && body.length > previewLines && (
                  <div className="cursor-trace-step-more">
                    … 还有 {body.length - previewLines} 行
                  </div>
                )}
              </div>
            )}
          </details>
        );
      })}
    </div>
  );
}
