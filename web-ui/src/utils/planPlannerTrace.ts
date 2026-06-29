import type { PlanDetail } from '../api/client';

/** 当前 Plan 的 Planner 版本号（修订后递增）。 */
export function planPlannerVersion(detail: PlanDetail | null | undefined): number {
  if (!detail) {
    return 1;
  }
  const fromState = detail.plan_state?.plan_version;
  if (typeof fromState === 'number' && fromState > 0) {
    return fromState;
  }
  const plan = detail.plan as { version?: number } | undefined;
  const fromPlan = plan?.version;
  if (typeof fromPlan === 'number' && fromPlan > 0) {
    return fromPlan;
  }
  return 1;
}

/** Planner 子 Agent 的 session thread_id。 */
export function planPlannerSubThread(planThread: string, version: number): string {
  return `${planThread}:planner:v${version || 1}`;
}
