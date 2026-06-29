import type { MessageItem, TreeNode } from '../../api/client';
import { parseApiMessagesToChat } from '../../utils/messageText';
import type { ChatMessage } from '../../components/console/ChatThread';
import type { TraceLine } from './types';
import type { TraceStep } from '../../types/trace';

function formatWorkerResultBlock(
  result: Record<string, unknown>,
  task: Record<string, unknown> | undefined,
  chat: ChatMessage[],
): string | null {
  const summary = String(result.summary || '').trim();
  const files = Array.isArray(result.files_changed)
    ? result.files_changed.map((f) => String(f)).filter(Boolean)
    : [];
  const readonly = Boolean(task?.readonly);
  const lastText = chat.length > 0 ? chat[chat.length - 1]?.text || '' : '';
  if (lastText.includes('### JSON 摘要') || (summary && lastText.includes(summary.slice(0, 80)))) {
    return null;
  }
  const lines: string[] = ['---', '**任务结果**'];
  if (summary) {
    lines.push('', summary);
  }
  if (files.length > 0) {
    lines.push('', '**改动文件：**');
    for (const f of files) {
      lines.push(`- \`${f}\``);
    }
  } else if (readonly) {
    lines.push('', '_本 Work 为只读任务（读取/检索），不会写文件。落盘请看 **w5** 等可写任务。_');
  } else {
    lines.push('', '_未检测到文件改动（Worker 可能仅完成规划未执行 write_file）。_');
  }
  return lines.join('\n');
}

import { splitAssistantPlanningLines } from '../../utils/messageText';

export function parseWorkerMessages(data: {
  messages: MessageItem[];
  result: Record<string, unknown> | null;
  task?: Record<string, unknown>;
}): {
  chat: ChatMessage[];
  traces: TraceLine[];
  traceSteps: TraceStep[];
} {
  const { chat: rawChat, toolTraces } = parseApiMessagesToChat(data.messages || [], {
    toolToTrace: true,
    idPrefix: 'w',
  });
  const chat: ChatMessage[] = [];
  const traceParts: TraceLine[] = toolTraces.map((t) => ({ id: t.id, text: t.text }));

  for (const m of rawChat) {
    if (m.role === 'thinking') {
      chat.push(m);
      continue;
    }
    if (m.role === 'trace') {
      traceParts.push({ id: m.id, text: m.text });
      continue;
    }
    if (m.role === 'assistant') {
      const { planLines, replyLines } = splitAssistantPlanningLines(m.text);
      if (planLines.length > 0) {
        traceParts.push({ id: `${m.id}-plan`, text: planLines.join('\n') });
      }
      const reply = replyLines.join('\n').trim();
      if (reply) {
        chat.push({ ...m, text: reply });
      }
      continue;
    }
    if (m.role === 'user') {
      const prev = chat[chat.length - 1];
      if (prev?.role === 'user' && prev.text === m.text) {
        continue;
      }
      chat.push(m);
      continue;
    }
    chat.push(m);
  }

  if (data.result) {
    const block = formatWorkerResultBlock(data.result, data.task, chat);
    if (block) {
      chat.push({
        id: 'result',
        role: 'assistant',
        text: block,
      });
    }
  }

  return { chat, traces: traceParts, traceSteps: [] };
}

export function findPlanNode(plans: TreeNode[], threadId: string): TreeNode | null {
  for (const plan of plans) {
    if (plan.thread_id === threadId) {
      return plan;
    }
  }
  return null;
}
