import type { MessageItem } from '../api/client';
import type { ChatImageAttachment } from '../types/chatImage';
import type { ChatMessage } from '../components/console/ChatThread';
import { stripSurveyForDisplay } from './surveyDisplay';

/**
 * 从 LangChain / 网关消息 content 提取用户可见正文（仅 text 块，忽略 thinking）。
 * 支持 JSON 与 Python repr 形式的 list 脏数据（与后端 normalize_stored_llm_text 对齐）。
 */
export function extractMessageContent(content: unknown): string {
  if (content == null) {
    return '';
  }
  if (typeof content === 'string') {
    const trimmed = content.trim();
    if (trimmed.startsWith('[') && (trimmed.includes('"type"') || trimmed.includes("'type'"))) {
      const fromJson = tryParseContentList(trimmed);
      if (fromJson) {
        return fromJson;
      }
    }
    return content;
  }
  if (Array.isArray(content)) {
    return extractFromContentBlocks(content).text;
  }
  return String(content);
}

function extractImagesFromContentBlocks(blocks: unknown[]): ChatImageAttachment[] {
  const images: ChatImageAttachment[] = [];
  for (const block of blocks) {
    if (!block || typeof block !== 'object') {
      continue;
    }
    const row = block as Record<string, unknown>;
    const blockType = String(row.type ?? '').toLowerCase();
    if (blockType === 'image_ref') {
      const id = String(row.id ?? '').trim();
      const mediaType = String(row.media_type ?? 'image/png');
      if (id) {
        images.push({ id, media_type: mediaType });
      }
    }
  }
  return images;
}

function extractFromContentBlocks(blocks: unknown[]): { text: string; images: ChatImageAttachment[] } {
  const parts: string[] = [];
  for (const block of blocks) {
    if (typeof block === 'string') {
      parts.push(block);
      continue;
    }
    if (!block || typeof block !== 'object') {
      continue;
    }
    const row = block as Record<string, unknown>;
    const kind = String(row.type ?? '').toLowerCase();
    if (kind === 'text') {
      parts.push(String(row.text ?? ''));
    }
  }
  return {
    text: parts.join('').trim(),
    images: extractImagesFromContentBlocks(blocks),
  };
}

function tryParseContentList(trimmed: string): string | null {
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (Array.isArray(parsed)) {
      const text = extractFromContentBlocks(parsed).text;
      return text || null;
    }
  } catch {
    /* JSON 失败，尝试 Python repr */
  }
  try {
    const jsonish = trimmed
      .replace(/\bNone\b/g, 'null')
      .replace(/\bTrue\b/g, 'true')
      .replace(/\bFalse\b/g, 'false')
      .replace(/'/g, '"');
    const parsed = JSON.parse(jsonish) as unknown;
    if (Array.isArray(parsed)) {
      const text = extractFromContentBlocks(parsed).text;
      return text || null;
    }
  } catch {
    /* ignore */
  }
  return null;
}

export function resolveHistoryImages(
  content: unknown,
  images?: ChatImageAttachment[],
): ChatImageAttachment[] {
  if (Array.isArray(images) && images.length > 0) {
    return images;
  }
  if (Array.isArray(content)) {
    return extractFromContentBlocks(content).images;
  }
  return [];
}
export function resolveHistoryDisplayText(
  content: unknown,
  displayText?: string,
  _raw?: Record<string, unknown>,
  _toolCalls?: unknown,
): string {
  if (displayText !== undefined && displayText !== null) {
    return String(displayText).trim();
  }
  return extractMessageContent(content).trim();
}

/** ReAct 规划/Trace 行，不应进聊天主区（与 Worker 一致）。 */
export function isPlanningLine(text: string): boolean {
  const t = text.trim();
  return (
    t.startsWith('【规划】') ||
    t.startsWith('▶') ||
    t.startsWith('│ 规划') ||
    /^│\s*└/.test(t)
  );
}

/** 从 messages.jsonl raw 读取 llgraph.thinking_text（模型内部推理，非可见正文）。 */
export function extractLlgraphThinkingText(raw?: Record<string, unknown>): string {
  if (!raw) {
    return '';
  }
  const data = (raw.data ?? raw) as Record<string, unknown>;
  if (!data || typeof data !== 'object') {
    return '';
  }
  const kwargs = data.additional_kwargs as Record<string, unknown> | undefined;
  if (!kwargs || typeof kwargs !== 'object') {
    return '';
  }
  const llgraph = kwargs.llgraph as Record<string, unknown> | undefined;
  if (!llgraph || typeof llgraph !== 'object') {
    return '';
  }
  const thinking = llgraph.thinking_text;
  return typeof thinking === 'string' ? thinking.trim() : '';
}

/** 按行拆分助手正文：规划/trace 行 vs 用户可见答复（代码块内不拆）。 */
export function splitAssistantPlanningLines(text: string): {
  planLines: string[];
  replyLines: string[];
} {
  const planLines: string[] = [];
  const replyLines: string[] = [];
  let inFence = false;
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('```')) {
      inFence = !inFence;
      replyLines.push(line);
      continue;
    }
    if (!inFence && isPlanningLine(line)) {
      planLines.push(line);
    } else {
      replyLines.push(line);
    }
  }
  return { planLines, replyLines };
}

/** 历史 tool 输出指针，不应作为 Trace 展示。 */
export function isArchivedToolPlaceholder(text: string): boolean {
  const t = text.trim();
  return (
    t.includes('[历史工具输出已省略') ||
    t.includes('[历史工具输出已归档]') ||
    t.includes('.llgraph/context/tool-results/')
  );
}

/** 从发给模型的 user 消息中剥离 workspace-context 等注入块。 */
export function stripInjectedContext(text: string): string {
  let out = text;
  out = out.replace(/<workspace-context>[\s\S]*?<\/workspace-context>\s*/gi, '');
  out = out.replace(/<session-manifest>[\s\S]*?<\/session-manifest>\s*/gi, '');
  out = out.replace(/<custom-command[\s\S]*?<\/custom-command>\s*/gi, '');
  return out.trim();
}

const PLAIN_FUNCTIONS_HEAD = /functions\.[A-Za-z0-9_]+:\d+/g;

function skipJsonObject(text: string, start: number): number {
  if (start >= text.length || text[start] !== '{') {
    return start;
  }
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let idx = start; idx < text.length; idx += 1) {
    const ch = text[idx];
    if (inString) {
      if (escape) {
        escape = false;
      } else if (ch === '\\') {
        escape = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return idx + 1;
      }
    }
  }
  return text.length;
}

/** 剥离 Kimi / plain functions / XML tool call markup（与后端 strip_inbound 对齐）。 */
export function stripInboundToolCallMarkup(text: string): string {
  if (!text) {
    return '';
  }
  let out = text;
  out = out.replace(/<\|(?:redacted_)?tool_calls_section_begin\|>[\s\S]*?<\|(?:redacted_)?tool_calls_section_end\|>/g, '');
  out = out.replace(
    /<\|(?:redacted_)?tool_call_begin(?:_kimi)?\|>[\s\S]*?<\|(?:redacted_)?tool_call_end(?:_kimi)?\|>/g,
    '',
  );
  out = out.replace(/<\|(?:redacted_)?tool_call_argument_begin\|>/g, '');
  out = out.replace(/<tool_call[\s\S]*?<\/tool_call>/gi, '');
  if (out.includes('functions.')) {
    const parts: string[] = [];
    let cursor = 0;
    while (cursor < out.length) {
      PLAIN_FUNCTIONS_HEAD.lastIndex = cursor;
      const match = PLAIN_FUNCTIONS_HEAD.exec(out);
      if (!match || match.index == null) {
        parts.push(out.slice(cursor));
        break;
      }
      parts.push(out.slice(cursor, match.index));
      let end = match.index + match[0].length;
      if (end < out.length && out[end] === '{') {
        end = skipJsonObject(out, end);
      }
      cursor = end;
    }
    out = parts.join('');
  }
  return out.replace(/\n{3,}/g, '\n\n').trim();
}

/** Web 聊天区助手正文：剥离 tool markup 与【规划】行。 */
export function formatAgentChatDisplayText(text: string): string {
  const cleaned = stripSurveyForDisplay(stripInboundToolCallMarkup(text || ''));
  const lines = cleaned
    .split('\n')
    .filter((ln) => ln.trim() && !ln.trim().startsWith('【规划】'));
  return lines.join('\n').trim();
}

export function isInjectedSystemContent(text: string): boolean {
  const t = text.trim();
  if (!t) {
    return true;
  }
  if (isThinkContinueNudge(t)) {
    return true;
  }
  if (t.includes('<session-manifest>') || t.includes('<workspace-context>')) {
    return true;
  }
  if (t.startsWith('Skills/Rules') && t.includes('manifest:')) {
    return true;
  }
  return false;
}

/** 与后端 agent_turn.THINK_CONTINUE_NUDGE 对齐 */
export function isThinkContinueNudge(text: string): boolean {
  const t = text.trim();
  if (!t) {
    return false;
  }
  return (
    t.includes('你上一轮仅在 thinking/reasoning 中推理') ||
    (t.startsWith('[系统]') && t.includes('thinking-only'))
  );
}

export const THINK_NUDGE_SUMMARY = '系统续跑（模型未输出可见正文）';

export function thinkNudgeDetailText(_raw?: string): string {
  return (
    '上一回复仅有内部推理、未在正文 text 中给出用户可见答复，Agent 自动续跑一轮。' +
    '此提示仅用于调试，不是用户输入。'
  );
}

/** 检索/排查过程中的中间笔记，不应作为最终助手答复展示。 */
export function isInterimInvestigationText(text: string): boolean {
  const t = formatAgentChatDisplayText(text).trim();
  if (!t || t.length < 48) {
    return false;
  }
  if (/^#{2,3}\s+/m.test(t)) {
    return false;
  }
  let score = 0;
  if (/根据(?:搜索|检索|查询)结果/.test(t)) {
    score += 2;
  }
  if (/目前找到(?:最接近|的相关)?/.test(t)) {
    score += 2;
  }
  if (/^(?:让我|接下来|需要再?(?:看|查|搜索|确认|继续))/m.test(t)) {
    score += 1;
  }
  const numbered = (t.match(/^\s*\d+[.．、)\)]\s+/gm) || []).length;
  if (numbered >= 3) {
    score += 2;
  }
  if (/(?:`[^`]+:\d+`|:\d+\s*$)/m.test(t)) {
    score += 1;
  }
  if (/未(?:直接)?(?:看到|找到)|尚未(?:确认|验证)/.test(t)) {
    score += 1;
  }
  return score >= 3;
}

function assistantMessageRole(
  visible: string,
  hasTools: boolean,
  afterThinkNudge: boolean,
): 'assistant' | 'thinking' | null {
  if (!visible.trim()) {
    return null;
  }
  if (hasTools) {
    return 'thinking';
  }
  if (afterThinkNudge && isInterimInvestigationText(visible)) {
    return 'thinking';
  }
  if (isInterimInvestigationText(visible)) {
    return 'thinking';
  }
  return 'assistant';
}

export interface ParseApiMessagesOptions {
  /** tool 消息是否进入 trace 列表（Worker 主区） */
  toolToTrace?: boolean;
  idPrefix?: string;
}

/**
 * 将 API messages.jsonl 条目解析为 ChatThread 消息（与 Agent loadHistory 一致）。
 */
export function parseApiMessagesToChat(
  messages: MessageItem[],
  options: ParseApiMessagesOptions = {},
): { chat: ChatMessage[]; toolTraces: Array<{ id: string; text: string }> } {
  const { toolToTrace = false, idPrefix = 'h' } = options;
  const chat: ChatMessage[] = [];
  const toolTraces: Array<{ id: string; text: string }> = [];

  let afterThinkNudge = false;

  for (let i = 0; i < messages.length; i += 1) {
    const m = messages[i];
    const display = resolveHistoryDisplayText(
      m.content,
      m.display_text,
      m.raw,
      m.tool_calls,
    );
    const type = String(m.type || '').toLowerCase();
    const msgKind = String((m as { kind?: string }).kind || '').toLowerCase();
    if (type.includes('system')) {
      continue;
    }
    if (type.includes('tool')) {
      if (toolToTrace && display.trim() && !isArchivedToolPlaceholder(display)) {
        toolTraces.push({ id: `wt-${i}`, text: display });
      }
      continue;
    }
    const isUser = type.includes('human') || type === 'user';
    if (isUser) {
      if (msgKind === 'think_nudge' || isThinkContinueNudge(display)) {
        chat.push({
          id: `${idPrefix}-${i}`,
          role: 'system',
          banner: 'nudge',
          text: thinkNudgeDetailText(display),
        });
        afterThinkNudge = true;
        continue;
      }
      const cleaned = stripInjectedContext(display);
      const images = resolveHistoryImages(m.content, (m as { images?: ChatImageAttachment[] }).images);
      if ((!cleaned && images.length === 0) || (cleaned && isInjectedSystemContent(cleaned))) {
        continue;
      }
      chat.push({ id: `${idPrefix}-${i}`, role: 'user', text: cleaned, images });
      afterThinkNudge = false;
    } else if (type.includes('ai') || type.includes('assistant')) {
      const visible = formatAgentChatDisplayText(display);
      const hasTools = Boolean(m.tool_calls && Array.isArray(m.tool_calls) && m.tool_calls.length > 0);
      const thinkingMeta = extractLlgraphThinkingText(m.raw);
      const role = assistantMessageRole(visible, hasTools, afterThinkNudge);
      if (role === 'thinking') {
        const thinkBody = thinkingMeta || visible;
        if (thinkBody.trim()) {
          chat.push({ id: `${idPrefix}-${i}`, role: 'thinking', text: thinkBody });
        }
        afterThinkNudge = false;
        continue;
      }
      if (role === 'assistant') {
        chat.push({ id: `${idPrefix}-${i}`, role: 'assistant', text: visible });
      }
      afterThinkNudge = false;
    } else if (display.trim() && !isInjectedSystemContent(display)) {
      const visible = formatAgentChatDisplayText(display);
      if (visible.trim()) {
        chat.push({ id: `${idPrefix}-${i}`, role: 'assistant', text: visible });
      }
    }
  }

  return { chat, toolTraces };
}

/**
 * Agent 历史：剥离【规划】/Trace 行，仅保留用户可见答复（对齐 Worker parseWorkerMessages）。
 */
export function parseAgentHistoryMessages(messages: MessageItem[]): ChatMessage[] {
  const { chat: rawChat } = parseApiMessagesToChat(messages);
  const chat: ChatMessage[] = [];
  for (const m of rawChat) {
    if (m.role === 'thinking') {
      chat.push(m);
      continue;
    }
    if (m.role === 'trace') {
      chat.push(m);
      continue;
    }
    if (m.role === 'assistant') {
      const { planLines, replyLines } = splitAssistantPlanningLines(m.text);
      const reply = formatAgentChatDisplayText(replyLines.join('\n'));
      if (planLines.length > 0) {
        chat.push({ id: `${m.id}-plan`, role: 'thinking', text: planLines.join('\n') });
      }
      if (reply.trim()) {
        chat.push({ ...m, text: reply });
      }
      continue;
    }
    chat.push(m);
  }
  return chat;
}

/** 去掉末尾与历史重复的用户消息（pending/trace 重进时在底部多追加一条）。 */
export function dedupeUserMessages(chat: ChatMessage[]): ChatMessage[] {
  if (chat.length < 2) {
    return chat;
  }
  const last = chat[chat.length - 1];
  if (last.role !== 'user') {
    return chat;
  }
  const t = last.text.trim();
  if (!t) {
    return chat;
  }
  for (let i = 0; i < chat.length - 1; i += 1) {
    const m = chat[i];
    if (m.role === 'user' && m.text.trim() === t) {
      return chat.slice(0, -1);
    }
  }
  return chat;
}

const TRACE_USER_MSG_RE = /▶\s*用户消息\s+(.+)$/;

/** 从 trace 日志行提取最近一条用户消息（trace 内为 120 字预览）。 */
export function extractLatestUserTextFromTraceLines(lines: string[]): string | null {
  let last: string | null = null;
  for (const line of lines) {
    const m = line.match(TRACE_USER_MSG_RE);
    if (m?.[1]?.trim()) {
      last = m[1].trim();
    }
  }
  return last;
}

export function userMessageAlreadyInChat(chat: ChatMessage[], candidate: string): boolean {
  const t = candidate.trim();
  if (!t) {
    return true;
  }
  const matches = (u: string) =>
    u === t || u.startsWith(t) || t.startsWith(u.slice(0, Math.min(u.length, 120)));
  for (let i = chat.length - 1; i >= 0; i -= 1) {
    const m = chat[i];
    if (m.role !== 'user') {
      continue;
    }
    const u = m.text.trim();
    if (matches(u)) {
      return true;
    }
  }
  return false;
}

/** 去掉相邻重复的用户消息（断线重进 / pending 合并遗留）；保留带图片的那条。 */
export function dedupeConsecutiveUserMessages(chat: ChatMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const m of chat) {
    if (m.role === 'user' && out.length > 0) {
      const prev = out[out.length - 1];
      if (prev.role === 'user' && prev.text.trim() === m.text.trim()) {
        if ((m.images?.length ?? 0) > 0 && (prev.images?.length ?? 0) === 0) {
          out[out.length - 1] = { ...prev, images: m.images };
        }
        continue;
      }
    }
    out.push(m);
  }
  return out;
}

/** 重载历史时保留同会话用户消息的本地预览图（image_ref 尚未落盘时）。 */
export function preserveUserMessageImages(
  cached: ChatMessage[] | undefined,
  loaded: ChatMessage[],
): ChatMessage[] {
  const prevWithImages = (cached ?? []).filter(
    (m) => m.role === 'user' && (m.images?.length ?? 0) > 0,
  );
  if (prevWithImages.length === 0) {
    return loaded;
  }
  const textMatches = (a: string, b: string) => {
    const left = a.trim();
    const right = b.trim();
    if (!left || !right) {
      return false;
    }
    return (
      left === right
      || left.startsWith(right)
      || right.startsWith(left.slice(0, Math.min(left.length, 120)))
    );
  };
  return loaded.map((m) => {
    if (m.role !== 'user' || (m.images?.length ?? 0) > 0) {
      return m;
    }
    for (const prev of prevWithImages) {
      if (textMatches(prev.text, m.text)) {
        return { ...m, images: prev.images };
      }
    }
    return m;
  });
}

/** 运行中会话：合并 API 历史与同会话内存快照（保留尚未落盘的尾部消息与 blob 预览）。 */
export function mergeRunningSessionMessages(
  cached: ChatMessage[] | undefined,
  loaded: ChatMessage[],
): ChatMessage[] {
  if (!cached?.length) {
    return loaded;
  }
  let merged = preserveUserMessageImages(cached, loaded);
  let startIdx = cached.length;
  for (let i = 0; i < cached.length; i += 1) {
    const cm = cached[i];
    if (cm.role === 'user' && !userMessageAlreadyInChat(merged, cm.text)) {
      startIdx = i;
      break;
    }
  }
  if (startIdx < cached.length) {
    merged = dedupeConsecutiveUserMessages([...merged, ...cached.slice(startIdx)]);
    merged = preserveUserMessageImages(cached, merged);
  }
  return merged;
}

export function mergeChatWithPendingUserMessages(
  chat: ChatMessage[],
  opts: { traceLines?: string[]; pendingText?: string | null; allowTraceUser?: boolean },
): ChatMessage[] {
  const allowTrace = opts.allowTraceUser !== false;
  const candidates = [
    opts.pendingText?.trim() || '',
    allowTrace ? extractLatestUserTextFromTraceLines(opts.traceLines || []) || '' : '',
  ].filter(Boolean);
  let next = chat;
  for (const text of candidates) {
    if (userMessageAlreadyInChat(next, text)) {
      continue;
    }
    next = [...next, { id: `u-pending-${Date.now()}-${Math.random()}`, role: 'user', text }];
  }
  return next;
}