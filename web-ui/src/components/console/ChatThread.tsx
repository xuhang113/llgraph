import type { TraceStep, TraceTurn } from '../../types/trace';
import { isHelpReport } from '../../utils/helpReport';
import HelpReportView from './HelpReportView';
import MarkdownView from './MarkdownView';
import type { ChatImageAttachment } from '../../types/chatImage';
import ChatImageStrip from './ChatImageStrip';
import TraceFold from './TraceFold';
import SystemNudgeFold from './SystemNudgeFold';
import ThinkingFold from './ThinkingFold';
import { formatAgentChatDisplayText, THINK_NUDGE_SUMMARY } from '../../utils/messageText';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'trace' | 'thinking';
  text: string;
  images?: ChatImageAttachment[];
  traceSteps?: TraceStep[];
  /** system 消息展示样式 */
  banner?: 'help' | 'default' | 'nudge';
}

type RenderItem =
  | { kind: 'message'; message: ChatMessage }
  | { kind: 'trace'; id: string; text: string; steps?: TraceStep[] }
  | { kind: 'thinking'; id: string; segments: string[] };

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
  let thinkingParts: string[] = [];

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

  const flushThinking = () => {
    if (thinkingParts.length === 0) {
      return;
    }
    items.push({
      kind: 'thinking',
      id: `thinking-block-${items.length}`,
      segments: [...thinkingParts],
    });
    thinkingParts = [];
  };

  for (const m of normalizeTraceOrder(messages)) {
    if (m.role === 'thinking') {
      if (m.text.trim()) {
        thinkingParts.push(m.text.trim());
      }
      continue;
    }
    if (m.role === 'trace') {
      flushThinking();
      if (m.text) {
        traceParts.push(m.text);
      }
      if (m.traceSteps?.length) {
        traceStepParts.push(...m.traceSteps);
      }
      continue;
    }
    if (m.role === 'system') {
      flushThinking();
      flushTrace();
      items.push({ kind: 'message', message: m });
      continue;
    }
    flushThinking();
    flushTrace();
    items.push({ kind: 'message', message: m });
  }
  flushThinking();
  flushTrace();
  return items;
}

interface Props {
  messages: ChatMessage[];
  liveTraceText: string;
  liveTraceSteps: TraceStep[];
  liveTraceTurns?: TraceTurn[];
  streamText: string;
  busy: boolean;
  historyLoading?: boolean;
  traceMode?: string;
}

export default function ChatThread({
  messages,
  liveTraceText,
  liveTraceSteps,
  liveTraceTurns = [],
  streamText,
  busy,
  historyLoading = false,
  traceMode = 'steps',
}: Props) {
  const items = groupMessages(messages);
  /** 仅执行中在对话区展示 live trace；完成后收起，改由 ThinkingFold + 右侧面板 */
  const showLiveTrace =
    busy && (liveTraceText.trim().length > 0 || liveTraceSteps.length > 0);

  if (historyLoading && messages.length === 0) {
    return (
      <div className="cursor-chat-thread">
        <div className="cursor-chat-empty">加载会话…</div>
      </div>
    );
  }

  return (
    <div className={`cursor-chat-thread${historyLoading ? ' is-history-loading' : ''}`}>
      {items.length === 0 && !showLiveTrace && !streamText && !busy && (
        <div className="cursor-chat-empty">发送消息开始对话</div>
      )}
      {items.map((item) => {
        if (item.kind === 'thinking') {
          return <ThinkingFold key={item.id} segments={item.segments} />;
        }
        if (item.kind === 'trace') {
          return (
            <TraceFold key={item.id} text={item.text} steps={item.steps ?? []} />
          );
        }
        const m = item.message;
        if (m.role === 'system') {
          if (m.banner === 'nudge') {
            return (
              <SystemNudgeFold key={m.id} summary={THINK_NUDGE_SUMMARY} detail={m.text} />
            );
          }
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
                {(m.images?.length ?? 0) > 0 && <ChatImageStrip images={m.images ?? []} />}
                {m.text.trim() && <div className="cursor-user-query">{m.text}</div>}
              </>
            )}
            {m.role === 'assistant' && formatAgentChatDisplayText(m.text).trim() && (
              <>
                <div className="cursor-msg-label">助手</div>
                <div className="cursor-agent-reply">
                  <MarkdownView content={formatAgentChatDisplayText(m.text)} />
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
          turns={liveTraceTurns}
          live={busy}
        />
      )}
      {streamText && formatAgentChatDisplayText(streamText).trim() && (
        <article className="cursor-msg cursor-msg--assistant cursor-msg--streaming">
          <div className="cursor-msg-label">助手</div>
          <div className="cursor-agent-reply">
            <MarkdownView content={formatAgentChatDisplayText(streamText)} />
          </div>
        </article>
      )}
      {busy && !streamText && !showLiveTrace && (
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
