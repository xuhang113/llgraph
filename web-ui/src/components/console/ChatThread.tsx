import type { TraceStep, TraceTurn } from '../../types/trace';
import { useCallback, useMemo, useRef, type RefObject, type ReactNode } from 'react';
import { isHelpReport } from '../../utils/helpReport';
import HelpReportView from './HelpReportView';
import MarkdownView from './MarkdownView';
import CopyButton from './CopyButton';
import type { ChatImageAttachment } from '../../types/chatImage';
import ChatUserQueryBlock from './ChatUserQueryBlock';
import TraceFold from './TraceFold';
import SystemNudgeFold from './SystemNudgeFold';
import ThinkingFold from './ThinkingFold';
import ChatFloatingUserPrompt from './ChatFloatingUserPrompt';
import ChatRoleAvatar from './ChatRoleAvatar';
import { useFloatingUserPrompt } from '../../hooks/useFloatingUserPrompt';
import {
  formatAgentChatDisplayText,
  formatPlanningFoldText,
  isDispatchPlaceholderText,
  isUserVisibleAssistantText,
  splitAssistantStreamDisplay,
  stripLeadingDispatchPlaceholders,
  THINK_NUDGE_SUMMARY,
} from '../../utils/messageText';

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

type ChatTurn = { id: string; items: RenderItem[] };

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

/** 一轮内保证：思考 / Trace 在助手回复之前（与时间线、右侧 Trace 一致）。 */
function orderTurnItems(items: RenderItem[]): RenderItem[] {
  if (items.length <= 1) {
    return items;
  }
  const users: RenderItem[] = [];
  const thinking: RenderItem[] = [];
  const traces: RenderItem[] = [];
  const assistants: RenderItem[] = [];
  const rest: RenderItem[] = [];
  for (const item of items) {
    if (item.kind === 'thinking') {
      thinking.push(item);
      continue;
    }
    if (item.kind === 'trace') {
      traces.push(item);
      continue;
    }
    if (item.kind === 'message' && item.message.role === 'user') {
      users.push(item);
      continue;
    }
    if (item.kind === 'message' && item.message.role === 'assistant') {
      assistants.push(item);
      continue;
    }
    rest.push(item);
  }
  return [...users, ...thinking, ...traces, ...assistants, ...rest];
}

/** 按用户轮次分组：一问一答为一组，组间留更大间距。 */
function splitTurnSections(items: RenderItem[]): { preamble: RenderItem[]; turns: ChatTurn[] } {
  const preamble: RenderItem[] = [];
  const turns: ChatTurn[] = [];

  for (const item of items) {
    const isUser = item.kind === 'message' && item.message.role === 'user';
    if (isUser) {
      turns.push({ id: item.message.id, items: [item] });
      continue;
    }
    if (turns.length > 0) {
      turns[turns.length - 1].items.push(item);
    } else {
      preamble.push(item);
    }
  }

  return {
    preamble,
    turns: turns.map((turn) => ({ ...turn, items: orderTurnItems(turn.items) })),
  };
}

interface Props {
  messages: ChatMessage[];
  liveTraceText: string;
  liveTraceSteps: TraceStep[];
  liveTraceTurns?: TraceTurn[];
  streamText: string;
  /** SSE thinking_delta 全文（流式推理） */
  liveThinking?: string;
  busy: boolean;
  historyLoading?: boolean;
  traceMode?: string;
  /** 主滚动容器，用于浮动用户问题条与跳转 */
  scrollRootRef?: RefObject<HTMLElement | null>;
  /** 拉 explore 子 Trace */
  slug?: string;
}

export default function ChatThread({
  messages,
  liveTraceText,
  liveTraceSteps,
  liveTraceTurns = [],
  streamText,
  liveThinking = '',
  busy,
  historyLoading = false,
  traceMode = 'steps',
  scrollRootRef,
  slug,
}: Props) {
  const streamDisplay = useMemo(
    () => splitAssistantStreamDisplay(streamText),
    [streamText],
  );
  const livePlanningText = useMemo(() => {
    const rawPlan = streamDisplay.planningText
      ? formatPlanningFoldText(streamDisplay.planningText)
      : '';
    const planClean = stripLeadingDispatchPlaceholders(rawPlan).trim();
    const planOk =
      Boolean(planClean) &&
      !isDispatchPlaceholderText(planClean) &&
      isUserVisibleAssistantText(planClean);
    const think = liveThinking.trim();
    const parts = [planOk ? planClean : '', think].filter(Boolean);
    return parts.join('\n\n');
  }, [streamDisplay.planningText, liveThinking]);

  const items = groupMessages(messages);
  const { preamble, turns } = splitTurnSections(items);
  const userElementRefs = useRef<Map<string, HTMLElement>>(new Map());
  const floatingUser = useFloatingUserPrompt(scrollRootRef, userElementRefs, messages);

  const registerUserRef = useCallback((id: string, el: HTMLElement | null) => {
    if (el) {
      userElementRefs.current.set(id, el);
      return;
    }
    userElementRefs.current.delete(id);
  }, []);

  /** 执行中：对话区展示与右侧同类的步骤链（TraceFold） */
  const showLiveTrace =
    liveTraceTurns.length > 0 ||
    liveTraceSteps.length > 0 ||
    Boolean(liveTraceText.trim());

  const renderChatItem = (item: RenderItem): ReactNode => {
    if (item.kind === 'thinking') {
      return <ThinkingFold key={item.id} segments={item.segments} />;
    }
    if (item.kind === 'trace') {
      return (
        <TraceFold key={item.id} text={item.text} steps={item.steps ?? []} slug={slug} />
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

    if (m.role === 'user') {
      return (
        <article
          key={m.id}
          ref={(el) => registerUserRef(m.id, el)}
          className="cursor-msg cursor-msg--user"
          data-chat-user-turn=""
          aria-label="用户消息"
        >
          <div className="cursor-msg-row">
            <ChatRoleAvatar role="user" />
            <div className="cursor-msg-body">
              <ChatUserQueryBlock
                text={m.text}
                images={m.images}
                collapsed={m.text.trim().length > 600}
                expandable={m.text.trim().length > 600}
                expandedMaxHeight="min(52vh, 480px)"
              />
            </div>
          </div>
        </article>
      );
    }

    if (m.role === 'assistant' && isUserVisibleAssistantText(m.text)) {
      return (
        <article
          key={m.id}
          className="cursor-msg cursor-msg--assistant"
          aria-label="助手回复"
        >
          <div className="cursor-msg-row">
            <ChatRoleAvatar role="assistant" />
            <div className="cursor-msg-body">
              <CopyButton
                text={formatAgentChatDisplayText(m.text)}
                title="复制回复"
                className="cursor-msg-copy cursor-msg-copy--assistant"
                iconOnly
              />
              <div className="cursor-agent-reply">
                <MarkdownView content={formatAgentChatDisplayText(m.text)} />
              </div>
            </div>
          </div>
        </article>
      );
    }

    return null;
  };

  if (historyLoading && messages.length === 0) {
    return (
      <div className="cursor-chat-column">
        <div className="cursor-chat-thread">
          <div className="cursor-chat-empty">加载会话…</div>
        </div>
      </div>
    );
  }

  const liveReplyText = streamDisplay.replyText;
  const showLivePlanning = busy && Boolean(livePlanningText);
  const showLiveReply =
    Boolean(liveReplyText) && isUserVisibleAssistantText(liveReplyText);

  const hasTail =
    showLiveTrace || showLivePlanning || showLiveReply || busy;

  return (
    <div className="cursor-chat-column">
      {scrollRootRef && (floatingUser.visible || floatingUser.messageId) && (
        <ChatFloatingUserPrompt
          visible={floatingUser.visible}
          messageId={floatingUser.messageId}
          text={floatingUser.text}
          images={floatingUser.images}
          onJump={() => {
            if (floatingUser.messageId) {
              floatingUser.scrollToUser(floatingUser.messageId);
            }
          }}
        />
      )}
      <div className={`cursor-chat-thread${historyLoading ? ' is-history-loading' : ''}`}>
        {items.length === 0 && !hasTail && (
          <div className="cursor-chat-empty">发送消息开始对话</div>
        )}

        {preamble.map((item) => renderChatItem(item))}

        {turns.map((turn) => (
          <section key={turn.id} className="cursor-chat-turn">
            {turn.items.map((item) => renderChatItem(item))}
          </section>
        ))}

        {hasTail && (
          <section className="cursor-chat-turn cursor-chat-turn--live">
            {showLiveTrace && (
              <TraceFold
                text={liveTraceText}
                steps={liveTraceSteps}
                turns={liveTraceTurns}
                live={busy}
                omitUserMessageMisc
                slug={slug}
              />
            )}
            {showLivePlanning && (
              <ThinkingFold segments={[]} liveText={livePlanningText} live />
            )}
            {showLiveReply && (
              <article className="cursor-msg cursor-msg--assistant cursor-msg--streaming" aria-label="助手回复">
                <div className="cursor-msg-row">
                  <ChatRoleAvatar role="assistant" />
                  <div className="cursor-msg-body">
                    <CopyButton
                      text={liveReplyText}
                      title="复制回复"
                      className="cursor-msg-copy cursor-msg-copy--assistant"
                      iconOnly
                    />
                    <div className="cursor-agent-reply">
                      <MarkdownView content={liveReplyText} streaming />
                    </div>
                  </div>
                </div>
              </article>
            )}
            {busy && !showLiveReply && !showLivePlanning && !showLiveTrace && (
              <div className="cursor-thinking-wrap">
                <div className="cursor-thinking" aria-label="思考中">
                  <span className="cursor-thinking-dot" />
                  <span className="cursor-thinking-dot" />
                  <span className="cursor-thinking-dot" />
                </div>
                <p className="cursor-thinking-hint">执行中…</p>
                {(traceMode === 'none' || traceMode === 'reply') && (
                  <p className="cursor-thinking-hint">
                    当前 trace 为 <strong>{traceMode}</strong>，过程可能隐藏；
                    {traceMode === 'none' ? ' 可切到 steps / all。' : ' 仅流式显示最终回复。'}
                  </p>
                )}
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
