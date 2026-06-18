import type { TraceStep } from '../../types/trace';
import { isHelpReport } from '../../utils/helpReport';
import HelpReportView from './HelpReportView';
import MarkdownView from './MarkdownView';
import TraceFold from './TraceFold';
import { stripSurveyForDisplay } from '../../utils/surveyDisplay';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'trace';
  text: string;
  traceSteps?: TraceStep[];
  /** system 消息展示样式 */
  banner?: 'help' | 'default';
}

type RenderItem =
  | { kind: 'message'; message: ChatMessage }
  | { kind: 'trace'; id: string; text: string; steps?: TraceStep[] };

/** 修正「助手在 trace 前」的历史错位（流式结束时曾先落盘正文）。 */
function normalizeTraceOrder(messages: ChatMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (let i = 0; i < messages.length; i += 1) {
    const m = messages[i];
    const next = messages[i + 1];
    if (m.role === 'assistant' && next?.role === 'trace') {
      out.push(next);
      out.push(m);
      i += 1;
      continue;
    }
    out.push(m);
  }
  return out;
}

function groupMessages(messages: ChatMessage[]): RenderItem[] {
  const items: RenderItem[] = [];
  let traceParts: string[] = [];
  let traceStepParts: TraceStep[] = [];

  const flushTrace = () => {
    if (traceParts.length === 0 && traceStepParts.length === 0) {
      return;
    }
    items.push({
      kind: 'trace',
      id: `trace-block-${items.length}`,
      text: traceParts.join('\n'),
      steps: traceStepParts.length > 0 ? [...traceStepParts] : undefined,
    });
    traceParts = [];
    traceStepParts = [];
  };

  for (const m of normalizeTraceOrder(messages)) {
    if (m.role === 'trace') {
      if (m.text) {
        traceParts.push(m.text);
      }
      if (m.traceSteps?.length) {
        traceStepParts.push(...m.traceSteps);
      }
      continue;
    }
    if (m.role === 'system') {
      flushTrace();
      items.push({ kind: 'message', message: m });
      continue;
    }
    flushTrace();
    items.push({ kind: 'message', message: m });
  }
  flushTrace();
  return items;
}

interface Props {
  messages: ChatMessage[];
  liveTraceText: string;
  liveTraceSteps: TraceStep[];
  liveThinkingText: string;
  streamText: string;
  busy: boolean;
  traceMode?: string;
}

export default function ChatThread({
  messages,
  liveTraceText,
  liveTraceSteps,
  liveThinkingText,
  streamText,
  busy,
  traceMode = 'steps',
}: Props) {
  const items = groupMessages(messages);
  const hasLiveTrace =
    liveTraceText.trim().length > 0 || liveTraceSteps.length > 0 || liveThinkingText.trim().length > 0;
  const showLiveTrace = hasLiveTrace;

  return (
    <div className="cursor-chat-thread">
      {items.length === 0 && !showLiveTrace && !streamText && !busy && (
        <div className="cursor-chat-empty">发送消息开始对话</div>
      )}
      {items.map((item) => {
        if (item.kind === 'trace') {
          return (
            <TraceFold key={item.id} text={item.text} steps={item.steps ?? []} />
          );
        }
        const m = item.message;
        if (m.role === 'system') {
          const isHelp = m.banner === 'help' || isHelpReport(m.text);
          return (
            <div
              key={m.id}
              className={`cursor-system-banner${isHelp ? ' cursor-system-banner--help' : ''}`}
            >
              {isHelp ? <HelpReportView content={m.text} /> : <MarkdownView content={m.text} />}
            </div>
          );
        }
        return (
          <article key={m.id} className={`cursor-msg cursor-msg--${m.role}`}>
            {m.role === 'user' && (
              <>
                <div className="cursor-msg-label">你</div>
                <div className="cursor-user-query">{m.text}</div>
              </>
            )}
            {m.role === 'assistant' && (
              <>
                <div className="cursor-msg-label">助手</div>
                <div className="cursor-agent-reply">
                  <MarkdownView content={stripSurveyForDisplay(m.text)} />
                </div>
              </>
            )}
          </article>
        );
      })}
      {showLiveTrace && (
        <TraceFold
          text={liveTraceText}
          steps={liveTraceSteps}
          liveThinking={liveThinkingText}
          live={busy}
        />
      )}
      {streamText && (
        <article className="cursor-msg cursor-msg--assistant cursor-msg--streaming">
          <div className="cursor-msg-label">助手</div>
          <div className="cursor-agent-reply">
            <MarkdownView content={stripSurveyForDisplay(streamText)} />
          </div>
        </article>
      )}
      {busy && !streamText && !hasLiveTrace && (
        <div className="cursor-thinking-wrap">
          <div className="cursor-thinking" aria-label="思考中">
            <span className="cursor-thinking-dot" />
            <span className="cursor-thinking-dot" />
            <span className="cursor-thinking-dot" />
          </div>
          {(traceMode === 'none' || traceMode === 'reply') && (
            <p className="cursor-thinking-hint">
              当前 trace 为 <strong>{traceMode}</strong>，右侧不展示过程；
              {traceMode === 'none' ? ' 可切到 steps / all 查看详情。' : ' 仅流式显示最终回复。'}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
