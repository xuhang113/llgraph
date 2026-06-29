import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import type { WorkflowNode, WorkflowTask } from '../api/client';
import {
  bezierPath,
  buildWorkEdges,
  depsAllSatisfied,
  isTaskRunnable,
  layerHeaderText,
  taskLayers,
  terminalTaskIds,
  type TaskMeta,
} from '../utils/workflowDag';

const STATUS_ICON: Record<string, string> = {
  done: '✓',
  running: '●',
  waiting: '⏸',
  failed: '✗',
  pending: '○',
  skipped: '⊘',
};

const STATUS_LABEL: Record<string, string> = {
  done: '已完成',
  running: '执行中',
  waiting: '等待',
  failed: '失败',
  pending: '待执行',
  skipped: '已跳过',
};

const PIPELINE_LABELS: Record<string, string> = {
  planner: '规划',
  confirm: '确认',
  supervisor: '调度',
  synthesize: '汇总',
};

interface Props {
  slug: string;
  threadId: string;
  nodes: WorkflowNode[];
  tasks: WorkflowTask[];
  planTasks?: Array<Record<string, unknown>>;
  synthesizeDependsOn?: string[];
  currentTaskId?: string | null;
  variant?: 'compact' | 'main';
  onTaskSelect?: (taskId: string) => void;
  onTaskStop?: (taskId: string) => void;
  onTaskRun?: (taskId: string) => void;
  planJobRunning?: boolean;
}

interface DagEdge {
  id: string;
  d: string;
  active: boolean;
}

function mergeTaskMeta(
  workflowTasks: WorkflowTask[],
  planTasks?: Array<Record<string, unknown>>,
): TaskMeta[] {
  const planById = new Map<string, Record<string, unknown>>();
  for (const t of planTasks || []) {
    const id = String(t.id || '');
    if (id) {
      planById.set(id, t);
    }
  }
  const toMeta = (id: string, wf?: WorkflowTask, plan?: Record<string, unknown>): TaskMeta => {
    const deps = plan?.depends_on ?? wf?.depends_on;
    return {
      id,
      title: String(plan?.title ?? wf?.title ?? id),
      status: String(wf?.worker_node_status || wf?.status || plan?.status || 'pending'),
      depends_on: Array.isArray(deps) ? deps.map(String) : [],
    };
  };
  const seen = new Set<string>();
  const merged: TaskMeta[] = [];
  for (const t of workflowTasks) {
    seen.add(t.id);
    merged.push(toMeta(t.id, t, planById.get(t.id)));
  }
  for (const [id, plan] of planById) {
    if (!seen.has(id)) {
      merged.push(toMeta(id, undefined, plan));
    }
  }
  return merged;
}

export default function WorkflowGraph({
  slug,
  threadId,
  nodes,
  tasks,
  planTasks,
  synthesizeDependsOn,
  currentTaskId,
  variant = 'compact',
  onTaskSelect,
  onTaskStop,
  onTaskRun,
  planJobRunning = false,
}: Props) {
  const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n.status]));
  const mergedTasks = useMemo(() => mergeTaskMeta(tasks, planTasks), [tasks, planTasks]);
  const layers = useMemo(() => taskLayers(mergedTasks), [mergedTasks]);
  const byId = useMemo(() => new Map(mergedTasks.map((t) => [t.id, t])), [mergedTasks]);
  const workEdges = useMemo(() => buildWorkEdges(mergedTasks), [mergedTasks]);
  const synthesizeDeps = useMemo(() => {
    if (synthesizeDependsOn && synthesizeDependsOn.length > 0) {
      return synthesizeDependsOn;
    }
    return mergedTasks.map((t) => t.id);
  }, [synthesizeDependsOn, mergedTasks]);
  const terminalIds = useMemo(() => terminalTaskIds(mergedTasks), [mergedTasks]);

  const dagRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef<Map<string, HTMLElement>>(new Map());
  const synthesizeRef = useRef<HTMLDivElement>(null);
  const [dagEdges, setDagEdges] = useState<DagEdge[]>([]);

  const registerNodeRef = useCallback((id: string, el: HTMLElement | null) => {
    if (el) {
      nodeRefs.current.set(id, el);
    } else {
      nodeRefs.current.delete(id);
    }
  }, []);

  const updateEdges = useCallback(() => {
    const container = dagRef.current;
    if (!container || variant !== 'main' || mergedTasks.length === 0) {
      setDagEdges([]);
      return;
    }
    const containerRect = container.getBoundingClientRect();
    const next: DagEdge[] = [];

    for (const edge of workEdges) {
      const fromEl = nodeRefs.current.get(edge.from);
      const toEl = nodeRefs.current.get(edge.to);
      const fromTask = byId.get(edge.from);
      const toTask = byId.get(edge.to);
      if (!fromEl || !toEl || !fromTask || !toTask) {
        continue;
      }
      const active = depsAllSatisfied(toTask, byId);
      next.push({
        id: `${edge.from}->${edge.to}`,
        d: bezierPath(fromEl.getBoundingClientRect(), toEl.getBoundingClientRect(), containerRect),
        active,
      });
    }

    const synthEl = synthesizeRef.current;
    if (synthEl) {
      const synthRect = synthEl.getBoundingClientRect();
      const synthActive = synthesizeDeps.every((id) => {
        const st = byId.get(id)?.status || 'pending';
        return st === 'done' || st === 'skipped';
      });
      for (const tid of terminalIds) {
        const fromEl = nodeRefs.current.get(tid);
        if (!fromEl) {
          continue;
        }
        next.push({
          id: `${tid}->synthesize`,
          d: bezierPath(fromEl.getBoundingClientRect(), synthRect, containerRect),
          active: synthActive,
        });
      }
    }

    setDagEdges(next);
  }, [variant, mergedTasks.length, workEdges, byId, synthesizeDeps, terminalIds]);

  useLayoutEffect(() => {
    updateEdges();
    const container = dagRef.current;
    if (!container) {
      return undefined;
    }
    const ro = new ResizeObserver(() => updateEdges());
    ro.observe(container);
    window.addEventListener('resize', updateEdges);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', updateEdges);
    };
  }, [updateEdges, layers, mergedTasks, nodeMap.synthesize]);

  const renderWorker = (t: TaskMeta, opts?: { withRef?: boolean; showDepsText?: boolean }) => {
    const isCurrent = currentTaskId === t.id;
    const runnable = isTaskRunnable(t, byId);
    const blocked =
      (t.depends_on?.length || 0) > 0 && !depsAllSatisfied(t, byId) && t.status === 'pending';
    const inner = (
      <>
        <span className="wf-icon">{STATUS_ICON[t.status] || '○'}</span>
        <span className="wf-worker-id">{t.id}</span>
        <span className="wf-worker-title">{t.title}</span>
        <span className={`wf-status-badge status-${t.status}`}>{STATUS_LABEL[t.status] || t.status}</span>
        {runnable && <span className="wf-runnable-badge">可执行</span>}
        {blocked && <span className="wf-blocked-badge">等待依赖</span>}
        {opts?.showDepsText && t.depends_on && t.depends_on.length > 0 && (
          <span className="wf-worker-deps">依赖: {t.depends_on.join(', ')}</span>
        )}
        {(onTaskStop || onTaskRun) && (
          <span className="wf-worker-actions">
            {t.status === 'running' && onTaskStop && (
              <button
                type="button"
                className="wf-action-btn wf-action-btn--stop"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onTaskStop(t.id);
                }}
              >
                停止
              </button>
            )}
            {onTaskRun &&
              !planJobRunning &&
              (t.status === 'pending' || t.status === 'failed' || t.status === 'skipped') && (
                <button
                  type="button"
                  className="wf-action-btn wf-action-btn--run"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onTaskRun(t.id);
                  }}
                >
                  执行
                </button>
              )}
          </span>
        )}
      </>
    );
    const stateClass = `${runnable ? ' is-runnable' : ''}${blocked ? ' is-blocked' : ''}${isCurrent ? ' is-current' : ''}`;

    const card =
      onTaskSelect ? (
        <button
          type="button"
          className={`wf-worker wf-worker-btn status-${t.status}${stateClass}`}
          onClick={() => onTaskSelect(t.id)}
        >
          {inner}
        </button>
      ) : (
        <Link
          to={`/workspace/${slug}/plan/${threadId}/worker/${t.id}`}
          className={`wf-worker status-${t.status}${stateClass}`}
        >
          {inner}
        </Link>
      );

    if (opts?.withRef) {
      return (
        <div
          key={t.id}
          className="wf-layer-cell"
          ref={(el) => registerNodeRef(t.id, el)}
        >
          {card}
        </div>
      );
    }

    return (
      <div key={t.id} className="wf-layer-cell">
        {card}
      </div>
    );
  };

  const showLayeredDag = variant === 'main' && mergedTasks.length > 0;

  return (
    <div className={`workflow-graph workflow-graph--light${variant === 'main' ? ' workflow-graph--main' : ''}`}>
      <div className="wf-pipeline">
        {['planner', 'confirm', 'supervisor'].map((id, i) => (
          <div key={id} className="wf-step">
            {i > 0 && <span className="wf-arrow">→</span>}
            <div className={`wf-node status-${nodeMap[id] || 'pending'}`}>
              <span className="wf-icon">{STATUS_ICON[nodeMap[id]] || '○'}</span>
              <span className="wf-label">{PIPELINE_LABELS[id] || id}</span>
              <span className={`wf-status-badge status-${nodeMap[id] || 'pending'}`}>
                {STATUS_LABEL[nodeMap[id]] || STATUS_LABEL.pending}
              </span>
            </div>
          </div>
        ))}
      </div>

      {mergedTasks.length > 0 && (
        <div className="wf-workers">
          <div className="wf-workers-title">
            Work 任务
            {onTaskSelect ? '（连线表示依赖；上层完成后下层可执行）' : ''}
          </div>

          {showLayeredDag ? (
            <div className="wf-dag" ref={dagRef}>
              <svg className="wf-dag-svg" aria-hidden>
                {dagEdges.map((edge) => (
                  <path
                    key={edge.id}
                    d={edge.d}
                    className={`wf-dag-edge${edge.active ? ' is-active' : ''}`}
                  />
                ))}
              </svg>

              <div className="wf-layer-stack">
                {layers.map((layer, layerIndex) => {
                  const header = layerHeaderText(layerIndex, layers.length);
                  return (
                    <section key={layerIndex} className="wf-layer-section">
                      <header className="wf-layer-header">
                        <span className="wf-layer-badge">L{layerIndex + 1}</span>
                        <span className="wf-layer-title">{header.title}</span>
                        <span className="wf-layer-hint">{header.hint}</span>
                      </header>
                      <div className="wf-layer-grid">
                        {layer.map((t) => renderWorker(t, { withRef: true }))}
                      </div>
                    </section>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="wf-worker-grid">
              {mergedTasks.map((t) => renderWorker(t, { showDepsText: true }))}
            </div>
          )}
        </div>
      )}

      <div className="wf-pipeline wf-tail">
        <span className="wf-arrow">→</span>
        <div
          ref={synthesizeRef}
          className={`wf-node wf-synthesize-node status-${nodeMap.synthesize || 'pending'}`}
        >
          <span className="wf-icon">{STATUS_ICON[nodeMap.synthesize] || '○'}</span>
          <span className="wf-label">{PIPELINE_LABELS.synthesize}</span>
          <span className={`wf-status-badge status-${nodeMap.synthesize || 'pending'}`}>
            {STATUS_LABEL[nodeMap.synthesize] || STATUS_LABEL.pending}
          </span>
          {!showLayeredDag && synthesizeDeps.length > 0 && (
            <span className="wf-worker-deps">依赖: {synthesizeDeps.join(', ')}</span>
          )}
          {showLayeredDag && (
            <span className="wf-worker-deps">全部 Work 完成后生成最终报告</span>
          )}
        </div>
      </div>
    </div>
  );
}
