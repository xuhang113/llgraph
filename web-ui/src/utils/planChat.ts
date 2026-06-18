import type { MessageItem, PlanDetail } from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import {
  extractMessageContent,
  isInjectedSystemContent,
  parseApiMessagesToChat,
  stripInjectedContext,
} from './messageText';

function addUniqueUser(msgs: ChatMessage[], seen: Set<string>, text: string, id: string) {
  const cleaned = stripInjectedContext(text).trim();
  if (!cleaned || isInjectedSystemContent(cleaned) || seen.has(cleaned)) {
    return;
  }
  seen.add(cleaned);
  msgs.push({ id, role: 'user', text: cleaned });
}

/**
 * 从 Plan 详情与 messages.jsonl 构建主区对话列表。
 */
export function buildPlanChatMessages(
  detail: PlanDetail,
  history: MessageItem[] = [],
): ChatMessage[] {
  const msgs: ChatMessage[] = [];
  const seen = new Set<string>();

  if (detail.goal) {
    addUniqueUser(msgs, seen, detail.goal, 'goal');
  }

  const userMessages = detail.plan_state?.user_messages;
  if (Array.isArray(userMessages)) {
    userMessages.forEach((um, i) => {
      if (typeof um === 'string') {
        addUniqueUser(msgs, seen, um, `plan-um-${i}`);
      }
    });
  }

  const revision = detail.plan_state?.revision_note;
  if (typeof revision === 'string' && revision.trim()) {
    addUniqueUser(msgs, seen, `【待处理修订】${revision}`, 'revision-pending');
  }

  const { chat: historyChat } = parseApiMessagesToChat(history, { idPrefix: 'h' });
  for (const m of historyChat) {
    if (m.role === 'user') {
      addUniqueUser(msgs, seen, m.text, m.id);
    } else {
      msgs.push(m);
    }
  }

  if (detail.final_report) {
    const report = extractMessageContent(detail.final_report).trim();
    if (report) {
      msgs.push({ id: 'report', role: 'assistant', text: report });
    }
  }

  return msgs;
}
