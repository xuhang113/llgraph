import type { TraceStep } from '../../types/trace';
import { partitionTraceMiscLines } from '../../types/trace';
import TraceStepList from './TraceStepList';

interface Props {
  text: string;
  steps?: TraceStep[];
  liveThinking?: string;
  /** 进行中：默认展开 */
  live?: boolean;
}

const THINKING_PREVIEW_CHARS = 12_000;

function miscLineClass(line: string): string {
  if (line.includes('思考中')) {
    return 'cursor-trace-misc cursor-trace-misc--thinking';
  }
  if (line.includes('用户消息')) {
    return 'cursor-trace-misc cursor-trace-misc--user';
  }
  if (line.includes('并行执行') || line.startsWith('▶')) {
    return 'cursor-trace-misc cursor-trace-misc--milestone';
  }
  if (line.includes('提示:') || line.includes('实时:')) {
    return 'cursor-trace-misc cursor-trace-misc--hint';
  }
  return 'cursor-trace-misc';
}

export default function TraceFold({ text, steps = [], liveThinking = '', live = false }: Props) {
  const miscLines = partitionTraceMiscLines(text.split('\n'));
  const stepCount = steps.length;
  const miscCount = miscLines.length;
  const thinkingChars = liveThinking.trim().length;
  const totalHint = stepCount + miscCount + (thinkingChars > 0 ? 1 : 0);

  const summary = live
    ? `▶ 执行过程${
        totalHint > 0
          ? `（${thinkingChars > 0 ? `thinking ${thinkingChars} 字` : stepCount > 0 ? `${stepCount} 步` : `${miscCount} 行`}）`
          : '…'
      }`
    : `▶ Trace${totalHint > 0 ? `（${stepCount > 0 ? `${stepCount} 步` : `${miscCount} 行`}，点击展开）` : ''}`;

  const hasStructured = stepCount > 0 || miscCount > 0 || thinkingChars > 0;
  const showRawFallback = !stepCount && text.trim();
  const thinkingPreview =
    thinkingChars > THINKING_PREVIEW_CHARS
      ? liveThinking.trim().slice(-THINKING_PREVIEW_CHARS)
      : liveThinking.trim();

  return (
    <details className="cursor-trace-fold" open={live}>
      <summary className="cursor-trace-fold-summary">{summary}</summary>
      <div className="cursor-trace-fold-content">
        {thinkingPreview && (
          <details className="cursor-trace-thinking" open={live}>
            <summary className="cursor-trace-thinking-summary">
              ◎ 模型思考（{thinkingChars} 字{live ? ' · 流式' : ''}）
            </summary>
            <pre className="cursor-trace-thinking-body">{thinkingPreview}</pre>
          </details>
        )}
        {miscLines.map((line, index) => (
          <div key={index} className={miscLineClass(line)}>
            {line}
          </div>
        ))}
        <TraceStepList steps={steps} defaultOpenLast={live} />
        {showRawFallback && !hasStructured && (
          <pre className="cursor-trace-fold-body">{text}</pre>
        )}
      </div>
    </details>
  );
}
