import type { PlanDetail } from '../../api/client';
import WorkflowGraph from '../WorkflowGraph';
import MarkdownView from './MarkdownView';
import { formatTime, phaseBadgeClass } from '../../utils/format';

interface Props {
  slug: string;
  planDetail: PlanDetail;
  busy: boolean;
  onTaskSelect: (taskId: string) => void;
  onPlanConfirm: () => void;
  onPlanContinue: () => void;
}

function phaseLabel(phase: string): string {
  const map: Record<string, string> = {
    planning: '规划中',
    awaiting_confirm: '待确认',
    executing: '执行中',
    completed: '已完成',
    cancelled: '已取消',
  };
  return map[phase] || phase;
}

export default function PlanMainPanel({
  slug,
  planDetail,
  busy,
  onTaskSelect,
  onPlanConfirm,
  onPlanContinue,
}: Props) {
  const snap = planDetail.workflow_snapshot;
  const wfTasks = snap?.tasks || [];
  const planTasks = planDetail.tasks || [];
  const statusOf = (taskId: string): string => {
    const wf = wfTasks.find((t) => t.id === taskId);
    const row = planTasks.find((t) => String(t.id) === taskId);
    return String(wf?.worker_node_status || wf?.status || row?.status || 'pending');
  };
  const taskIds =
    planTasks.length > 0
      ? planTasks.map((t) => String(t.id || ''))
      : wfTasks.map((t) => t.id);
  const doneCount = taskIds.filter((id) => statusOf(id) === 'done').length;
  const totalCount = taskIds.length;
  const incompleteCount = taskIds.filter((id) => {
    const s = statusOf(id);
    return s === 'pending' || s === 'running' || s === 'failed';
  }).length;
  const currentTaskId = (snap as { current_task_id?: string })?.current_task_id || null;
  const showContinue =
    planDetail.phase === 'executing' && incompleteCount > 0 && !planDetail.job?.running;
  const showSynthesizeContinue =
    planDetail.phase === 'executing' &&
    incompleteCount === 0 &&
    !planDetail.final_report &&
    !planDetail.job?.running;

  return (
    <div className="cursor-plan-main">
      <section className="cursor-plan-main-summary">
        <div className="cursor-plan-main-meta">
          <span className={phaseBadgeClass(planDetail.phase)}>{phaseLabel(planDetail.phase)}</span>
          {totalCount > 0 && (
            <span className="cursor-plan-main-progress">
              Work {doneCount}/{totalCount}
            </span>
          )}
          {planDetail.job?.running && <span className="badge badge-running">后台运行中</span>}
          {planDetail.updated_at && (
            <span className="cursor-plan-main-updated">更新 {formatTime(planDetail.updated_at)}</span>
          )}
        </div>
        {planDetail.goal && (
          <p className="cursor-plan-main-goal">{planDetail.goal}</p>
        )}
        {planDetail.phase === 'completed' && (
          <p className="cursor-plan-main-hint">Plan 已终止。在输入框追问最终报告；重新规划请用 /plan revise。</p>
        )}
        {planDetail.error && (
          <div className="cursor-plan-main-error">{String(planDetail.error)}</div>
        )}
      </section>

      <section className="cursor-plan-main-workflow">
        <h2 className="cursor-plan-main-section-title">工作流</h2>
        <WorkflowGraph
          slug={slug}
          threadId={planDetail.thread_id}
          nodes={snap?.nodes || []}
          tasks={wfTasks}
          planTasks={planDetail.tasks}
          synthesizeDependsOn={snap?.synthesize_depends_on}
          currentTaskId={currentTaskId}
          variant="main"
          onTaskSelect={onTaskSelect}
        />
        <div className="cursor-plan-actions">
          {planDetail.phase === 'awaiting_confirm' && (
            <button type="button" className="cursor-btn-primary" onClick={onPlanConfirm}>
              确认计划
            </button>
          )}
          {showContinue && (
            <button
              type="button"
              className="cursor-btn-primary"
              onClick={onPlanContinue}
              disabled={busy}
            >
              继续执行未完成（已成功跳过）
            </button>
          )}
          {showSynthesizeContinue && (
            <button
              type="button"
              className="cursor-btn-primary"
              onClick={onPlanContinue}
              disabled={busy}
            >
              继续汇总
            </button>
          )}
          {!showContinue && !showSynthesizeContinue && planDetail.phase === 'executing' && incompleteCount === 0 && (
            <button
              type="button"
              className="cursor-btn-ghost"
              onClick={onPlanContinue}
              disabled={busy || !!planDetail.job?.running}
            >
              Continue
            </button>
          )}
        </div>
      </section>

      {planDetail.phase === 'completed' && planDetail.final_report && (
        <section className="cursor-plan-main-report">
          <h2 className="cursor-plan-main-section-title">最终报告</h2>
          <div className="cursor-plan-main-report-body">
            <MarkdownView content={planDetail.final_report} />
          </div>
        </section>
      )}
    </div>
  );
}
