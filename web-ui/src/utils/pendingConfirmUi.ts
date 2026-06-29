import type { Dispatch, MutableRefObject, SetStateAction } from 'react';
import type { PlanDetail, SurveySpec } from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import { api } from '../api/client';
import { buildPlanConfirmPayload } from '../pages/console/planHelpers';
import { extractMessageContent } from './messageText';
import {
  enqueueConfirm,
  peekConfirmHead,
  peekConfirmQueue,
  type PendingConfirmItem,
} from './pendingConfirmQueue';

export type PendingConfirmUiSetters = {
  setSurvey: Dispatch<SetStateAction<SurveySpec | null>>;
  setPlanConfirm: Dispatch<SetStateAction<Record<string, unknown> | null>>;
  setTaskStepConfirm: Dispatch<SetStateAction<string | null>>;
};

export function ingestSurveyConfirm(
  slug: string,
  threadId: string,
  survey: SurveySpec,
  assistantMessageId?: string,
): PendingConfirmItem {
  return enqueueConfirm(slug, threadId, {
    kind: 'survey',
    payload: survey,
    assistantMessageId,
  });
}

export function ingestPlanConfirmPayload(
  slug: string,
  threadId: string,
  payload: Record<string, unknown>,
): PendingConfirmItem {
  return enqueueConfirm(slug, threadId, {
    kind: 'plan_confirm',
    payload,
  });
}

export function ingestPlanConfirmFromDetail(
  slug: string,
  threadId: string,
  detail: PlanDetail,
): PendingConfirmItem {
  return ingestPlanConfirmPayload(slug, threadId, buildPlanConfirmPayload(detail));
}

export function ingestTaskStepConfirm(
  slug: string,
  threadId: string,
  taskId: string,
): PendingConfirmItem {
  return enqueueConfirm(slug, threadId, {
    kind: 'task_step_confirm',
    payload: { task_id: taskId },
  });
}

export function ingestPendingInterruptFromPlan(
  slug: string,
  threadId: string,
  detail: PlanDetail,
): void {
  const pending = detail.plan_state?.pending_interrupt;
  if (!pending || typeof pending !== 'object') {
    return;
  }
  const typ = String(pending.type || '');
  if (typ === 'plan_confirm') {
    ingestPlanConfirmPayload(slug, threadId, pending as Record<string, unknown>);
  } else if (typ === 'task_step_confirm') {
    ingestTaskStepConfirm(slug, threadId, String(pending.task_id || ''));
  }
}

/** 将队头待确认项同步到 UI state（modal / banner 数据源）。 */
export function applyPendingConfirmHead(
  slug: string,
  threadId: string,
  setters: PendingConfirmUiSetters,
  opts?: {
    surveyDismissedId?: string | null;
    taskStepDismissedId?: string | null;
  },
): PendingConfirmItem | null {
  const head = peekConfirmHead(slug, threadId);
  if (!head) {
    return null;
  }
  if (head.kind === 'survey') {
    if (opts?.surveyDismissedId && opts.surveyDismissedId === head.id) {
      return head;
    }
    setters.setSurvey(head.payload as SurveySpec);
    return head;
  }
  if (head.kind === 'plan_confirm') {
    setters.setPlanConfirm(head.payload as Record<string, unknown>);
    return head;
  }
  if (head.kind === 'task_step_confirm') {
    if (opts?.taskStepDismissedId && opts.taskStepDismissedId === head.id) {
      return head;
    }
    const taskId = String((head.payload as { task_id?: string })?.task_id || '');
    if (taskId) {
      setters.setTaskStepConfirm(taskId);
    }
    return head;
  }
  return head;
}

function lastAssistantRawText(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.role === 'assistant') {
      return extractMessageContent(msg.text);
    }
  }
  return '';
}

/** loadHistory 后：从队列 / 消息 / plan pending_interrupt 恢复待确认态。 */
export async function restorePendingConfirmsFromHistory(opts: {
  slug: string;
  threadId: string;
  messages?: ChatMessage[];
  planDetail?: PlanDetail | null;
  setters: PendingConfirmUiSetters;
  surveyDismissedRef?: MutableRefObject<string | null>;
  taskStepDismissedRef?: MutableRefObject<string | null>;
}): Promise<void> {
  const {
    slug,
    threadId,
    messages,
    planDetail,
    setters,
    surveyDismissedRef,
    taskStepDismissedRef,
  } = opts;

  if (planDetail) {
    ingestPendingInterruptFromPlan(slug, threadId, planDetail);
  }

  if (peekConfirmQueue(slug, threadId).length === 0 && messages && messages.length > 0) {
    const raw = lastAssistantRawText(messages);
    if (raw.trim()) {
      try {
        const { survey } = await api.resolveSurvey(slug, raw);
        if (survey) {
          const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
          ingestSurveyConfirm(slug, threadId, survey, lastAssistant?.id);
        }
      } catch {
        /* resolve 失败则跳过 */
      }
    }
  }

  applyPendingConfirmHead(slug, threadId, setters, {
    surveyDismissedId: surveyDismissedRef?.current ?? null,
    taskStepDismissedId: taskStepDismissedRef?.current ?? null,
  });
}
