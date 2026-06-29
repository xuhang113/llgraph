import type { PlanDetail } from '../../api/client';

export function planExecutionAllowWrite(detail: PlanDetail): boolean {
  const plan = detail.plan as { execution?: { allow_worker_write?: boolean } } | undefined;
  if (plan?.execution?.allow_worker_write) {
    return true;
  }
  const state = detail.plan_state as { allow_worker_write?: boolean } | undefined;
  return Boolean(state?.allow_worker_write);
}

export function planNeedsConfirm(detail: PlanDetail): boolean {
  const pending = detail.plan_state?.pending_interrupt;
  if (pending && typeof pending === 'object' && pending.type === 'plan_confirm') {
    return true;
  }
  if (detail.phase !== 'awaiting_confirm') {
    return false;
  }
  return !detail.job?.running;
}

export function buildPlanConfirmPayload(detail: PlanDetail): Record<string, unknown> {
  return {
    type: 'plan_confirm',
    title: detail.title,
    task_count: detail.tasks.length,
    error: detail.error || undefined,
    tasks: detail.tasks.map((t) => ({
      id: String(t.id || ''),
      title: String(t.title || t.id || ''),
    })),
  };
}
