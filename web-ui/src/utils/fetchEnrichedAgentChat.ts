/** 拉取 Agent 历史并用 session trace / live panel 补全助手正文（避免仅 messages.jsonl 丢回复）。 */

import { api } from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import type { TraceStep, TraceTurn } from '../types/trace';
import { mergeTraceStepsUnique, stepsToPanelLogLines } from '../types/trace';
import { hydrateChatMessageImages } from './chatAttachmentUrl';
import {
  dedupeConsecutiveUserMessages,
  finalizeAgentHistoryChat,
  parseAgentHistoryMessages,
  reconcileHistoryAfterTurnDone,
} from './messageText';
import { loadTracePanelCache } from './tracePanelStore';
import {
  loadTraceTurnsFromRemote,
  panelLinesFromTexts,
  parseTraceSteps,
  parseTraceTurnsFromRemote,
  preferRicherTraceCache,
  restorePanelTraceFromMessages,
} from '../pages/console/traceUtils';
import type { TraceLine } from '../pages/console/types';

export type EnrichAgentChatOpts = {
  fallbackReply?: string;
  liveLines?: TraceLine[];
  liveSteps?: TraceStep[];
  liveTurns?: TraceTurn[];
};

function assistantVisibleScore(chat: ChatMessage[]): number {
  return chat
    .filter((m) => m.role === 'assistant')
    .reduce((n, m) => n + (m.text?.trim().length || 0), 0);
}

function userMediaScore(chat: ChatMessage[]): number {
  return chat
    .filter((m) => m.role === 'user')
    .reduce((n, m) => n + (m.images?.length ?? 0) * 10_000 + (m.text?.trim().length || 0), 0);
}

function replyBodyScore(steps: TraceStep[]): number {
  return steps
    .filter((s) => s.kind === 'reply')
    .reduce((n, s) => n + (s.body_lines ?? []).join('\n').length, 0);
}

function turnsReplyBodyScore(turns: TraceTurn[]): number {
  return turns.reduce((n, t) => n + replyBodyScore(t.steps), 0);
}

/** 保留更完整的对话（助手正文 + 用户附图），防止瘦历史盖掉已展示内容。 */
export function preferRicherAgentChat(
  prev: ChatMessage[],
  next: ChatMessage[],
): ChatMessage[] {
  if (!prev.length) {
    return next;
  }
  if (!next.length) {
    return prev;
  }
  const nextAsst = assistantVisibleScore(next);
  const prevAsst = assistantVisibleScore(prev);
  if (nextAsst !== prevAsst) {
    return nextAsst >= prevAsst ? next : prev;
  }
  return userMediaScore(next) >= userMediaScore(prev) ? next : prev;
}

/**
 * 从 API messages + session trace（及可选 live panel）组装对话区消息。
 */
export async function fetchEnrichedAgentChatMessages(
  slug: string,
  threadId: string,
  opts: EnrichAgentChatOpts = {},
): Promise<ChatMessage[]> {
  const data = await api.messages(slug, threadId);
  const parsed = parseAgentHistoryMessages(data.messages || []);

  let panelLines: TraceLine[] = opts.liveLines?.length ? [...opts.liveLines] : [];
  let panelSteps: TraceStep[] = opts.liveSteps?.length ? [...opts.liveSteps] : [];
  let panelTurns: TraceTurn[] = opts.liveTurns?.length ? [...opts.liveTurns] : [];

  try {
    const remote = await api.sessionTrace(slug, threadId);
    const loaded = loadTraceTurnsFromRemote(remote);
    const remoteSteps = parseTraceSteps(remote.steps);
    // 同长度时也要比 reply body：liveTurns 可能缺 body_lines
    if (
      loaded.completed.length > panelTurns.length ||
      (loaded.completed.length === panelTurns.length &&
        turnsReplyBodyScore(loaded.completed) > turnsReplyBodyScore(panelTurns))
    ) {
      panelTurns = loaded.completed;
    }
    if (loaded.currentSteps.length > panelSteps.length) {
      panelSteps = loaded.currentSteps;
    }
    // completed 会话 currentSteps 为空，合并视图 steps 仍含各轮 reply
    if (remoteSteps.length > 0) {
      panelSteps = mergeTraceStepsUnique(panelSteps, remoteSteps);
    }
    if (panelTurns.length > 0) {
      panelSteps = mergeTraceStepsUnique(
        panelSteps,
        panelTurns.flatMap((t) => t.steps),
      );
    }
    if (remote.log_lines?.length && panelLines.length === 0) {
      panelLines = panelLinesFromTexts(remote.log_lines);
    } else if (panelLines.length === 0 && panelSteps.length > 0) {
      panelLines = stepsToPanelLogLines(panelSteps);
    }
  } catch {
    /* 无落盘 trace */
  }

  const picked = preferRicherTraceCache(slug, threadId, panelLines, panelSteps);
  panelLines = picked.lines;
  panelSteps = picked.steps;

  if (panelLines.length === 0 && panelSteps.length === 0) {
    const cached = loadTracePanelCache(slug, threadId);
    if (cached) {
      panelLines = panelLinesFromTexts(cached.log_lines);
      panelSteps = cached.steps;
    } else {
      const fromChat = restorePanelTraceFromMessages(parsed);
      panelLines = fromChat.lines;
      panelSteps = fromChat.steps;
    }
  }

  let traceTurnsForEnrich = panelTurns;
  let traceStepsForEnrich = panelSteps;
  if (traceTurnsForEnrich.length === 0 && traceStepsForEnrich.length > 0) {
    traceTurnsForEnrich = parseTraceTurnsFromRemote(undefined, traceStepsForEnrich);
    // 保留 steps，供 enrich 双通道补全
  }

  let chat = finalizeAgentHistoryChat(parsed, {
    traceLines: panelLines.map((l) => l.text),
    traceTurns: traceTurnsForEnrich,
    traceSteps: traceStepsForEnrich,
  });
  if (opts.fallbackReply) {
    chat = reconcileHistoryAfterTurnDone(chat, opts.fallbackReply);
  }
  chat = dedupeConsecutiveUserMessages(chat);
  return hydrateChatMessageImages(chat, slug, threadId);
}
