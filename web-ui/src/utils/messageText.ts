import type { MessageItem } from '../api/client';
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
    return extractFromContentBlocks(content);
  }
  return String(content);
}

function extractFromContentBlocks(blocks: unknown[]): string {
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
  return parts.join('').trim();
}

function tryParseContentList(trimmed: string): string | null {
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (Array.isArray(parsed)) {
      const text = extractFromContentBlocks(parsed);
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
      const text = extractFromContentBlocks(parsed);
      return text || null;
    }
  } catch {
    /* ignore */
  }
  return null;
}

/** 历史消息展示文本（含无正文 assistant 的 thinking 降级）。 */
export function resolveHistoryDisplayText(
  content: unknown,
  displayText?: string,
  raw?: Record<string, unknown>,
  toolCalls?: unknown,
): string {
  const fromField =
    typeof displayText === 'string' && displayText.trim() ? displayText.trim() : '';
  const fromContent = extractMessageContent(content).trim();
  let text = fromField || fromContent;
  if (!text && raw) {
    const data = (raw.data as Record<string, unknown> | undefined) ?? raw;
    const calls = toolCalls ?? data.tool_calls;
    const hasTools = Array.isArray(calls) && calls.length > 0;
    if (!hasTools) {
      const ak = data.additional_kwargs as Record<string, unknown> | undefined;
      const meta = ak?.llgraph as Record<string, unknown> | undefined;
      const thinking = meta?.thinking_text;
      if (typeof thinking === 'string' && thinking.trim()) {
        text = thinking.trim();
      }
    }
  }
  return text;
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

export function isInjectedSystemContent(text: string): boolean {
  const t = text.trim();
  if (!t) {
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

  for (let i = 0; i < messages.length; i += 1) {
    const m = messages[i];
    const display = resolveHistoryDisplayText(
      m.content,
      m.display_text,
      m.raw,
      m.tool_calls,
    );
    const type = String(m.type || '').toLowerCase();
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
      const cleaned = stripInjectedContext(display);
      if (!cleaned || isInjectedSystemContent(cleaned)) {
        continue;
      }
      chat.push({ id: `${idPrefix}-${i}`, role: 'user', text: cleaned });
    } else if (type.includes('ai') || type.includes('assistant')) {
      const visible = stripSurveyForDisplay(display);
      if (visible.trim()) {
        chat.push({ id: `${idPrefix}-${i}`, role: 'assistant', text: visible });
      }
    } else if (display.trim() && !isInjectedSystemContent(display)) {
      const visible = stripSurveyForDisplay(display);
      if (visible.trim()) {
        chat.push({ id: `${idPrefix}-${i}`, role: 'assistant', text: visible });
      }
    }
  }

  return { chat, toolTraces };
}
