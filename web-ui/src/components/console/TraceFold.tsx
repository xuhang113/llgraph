import type { TraceStep, TraceTurn } from '../../types/trace';
import {
  filterTraceMiscWhenSteps,
  filterTraceStepsForDisplay,
  partitionTraceMiscLines,
} from '../../types/trace';
import TraceStepList from './TraceStepList';
import TraceTurnList from './TraceTurnList';

interface Props {
  text: string;
  steps?: TraceStep[];
  turns?: TraceTurn[];
  liveThinking?: string;
  /** 进行中：默认展开 */
  live?: boolean;
  /** 对话区 live：用户气泡已展示，隐藏日志里的「用户消息」行 */
  omitUserMessageMisc?: boolean;
  /** 拉 explore 子 Trace */
  slug?: string;
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

export default function TraceFold({
  text,
  steps = [],
  turns = [],
  liveThinking = '',
  live = false,
  omitUserMessageMisc = false,
  slug,
}: Props) {
  const miscLines = partitionTraceMiscLines(text.split('\n'));
  const stepCount = filterTraceStepsForDisplay(steps).length;
  const turnCount = turns.length;
  const turnStepCount = turns.reduce(
    (sum, turn) => sum + filterTraceStepsForDisplay(turn.steps).length,
    0,
  );
  const displayMisc = filterTraceMiscWhenSteps(miscLines, stepCount).filter(
    (line) => !(omitUserMessageMisc && line.includes('用户消息')),
  );
  const miscCount = displayMisc.length;
  const thinkingChars = liveThinking.trim().length;
  const totalSteps = turnCount > 0 ? turnStepCount : stepCount;
  const totalHint = totalSteps + miscCount + (thinkingChars > 0 ? 1 : 0);

  const summary = live
    ? `▶ 执行过程${
        totalHint > 0
          ? `（${thinkingChars > 0 ? `thinking ${thinkingChars} 字` : totalSteps > 0 ? `${totalSteps} 步` : `${miscCount} 行`}）`
          : '…'
      }`
    : `▶ Trace${totalHint > 0 ? `（${totalSteps > 0 ? `${totalSteps} 步` : `${miscCount} 行`}，点击展开）` : ''}`;

  const hasStructured = totalSteps > 0 || miscCount > 0 || thinkingChars > 0;
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
        {displayMisc.length > 0 && (
          <div className="cursor-trace-misc-block">
            {displayMisc.map((line, index) => (
              <div key={index} className={miscLineClass(line)}>
                {line}
              </div>
            ))}
          </div>
        )}
        {turnCount > 0 ? (
          <TraceTurnList turns={turns} defaultOpenLast={live} slug={slug} />
        ) : (
          stepCount > 0 && (
            <div className="cursor-trace-steps">
              <div className="cursor-trace-steps-label">步骤（可展开详情）</div>
              <TraceStepList steps={steps} defaultOpenLast={live} slug={slug} />
            </div>
          )
        )}
        {showRawFallback && !hasStructured && (
          <pre className="cursor-trace-fold-body">{text}</pre>
        )}
      </div>
    </details>
  );
}
