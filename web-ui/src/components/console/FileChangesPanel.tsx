import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  api,
  type FileChangeItem,
  type FileChangesSummary,
  type PlanFileChangesSummary,
} from '../../api/client';
import { useAppDialog } from '../AppDialog';

export type FileChangesMode = 'agent' | 'plan' | 'worker';

interface Props {
  slug: string;
  mode: FileChangesMode;
  /** Agent / Worker 的 session thread */
  sessionThreadId: string;
  /** Plan 主 thread（mode=plan 时） */
  planThreadId?: string;
  /** Worker task id（mode=worker 时用于单 Work 回滚） */
  taskId?: string;
  busy: boolean;
  allowWrite?: boolean;
  /** 嵌入输入框 dock 顶栏（Cursor 风格） */
  embedded?: boolean;
  onStop?: () => void;
  onChangesUpdated?: () => void;
}

type PlanFileChangeItem = FileChangeItem & { task_id?: string; thread_id?: string };

export default function FileChangesPanel({
  slug,
  mode,
  sessionThreadId,
  planThreadId,
  taskId,
  busy,
  embedded = false,
  onStop,
  onChangesUpdated,
}: Props) {
  const { alert, confirm } = useAppDialog();
  const [expanded, setExpanded] = useState(false);
  const [agentChanges, setAgentChanges] = useState<FileChangesSummary | null>(null);
  const [planChanges, setPlanChanges] = useState<PlanFileChangesSummary | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [diffText, setDiffText] = useState('');
  const [diffLoading, setDiffLoading] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const [reviewHint, setReviewHint] = useState('');

  const refresh = useCallback(async () => {
    if (!slug) {
      return;
    }
    try {
      if (mode === 'plan' && planThreadId) {
        const data = await api.planFileChanges(slug, planThreadId);
        setPlanChanges(data);
        setAgentChanges(null);
      } else if (sessionThreadId) {
        const data = await api.fileChanges(slug, sessionThreadId);
        setAgentChanges(data);
        setPlanChanges(null);
      }
    } catch {
      setAgentChanges(null);
      setPlanChanges(null);
    }
  }, [slug, mode, planThreadId, sessionThreadId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const totalCount = mode === 'plan' ? planChanges?.total ?? 0 : agentChanges?.total ?? 0;
  const canUndo = mode === 'plan' ? Boolean(planChanges?.can_undo) : Boolean(agentChanges?.can_undo);

  const flatItems = useMemo((): PlanFileChangeItem[] => {
    if (mode === 'plan' && planChanges) {
      const items: PlanFileChangeItem[] = [];
      for (const group of planChanges.groups) {
        for (const p of group.paths) {
          items.push({ ...p, task_id: group.task_id, thread_id: group.thread_id });
        }
      }
      return items;
    }
    return (agentChanges?.paths ?? []) as PlanFileChangeItem[];
  }, [mode, planChanges, agentChanges]);

  const loadDiff = async (path: string, threadId: string) => {
    if (!slug) {
      return;
    }
    setSelectedPath(path);
    setDiffLoading(true);
    try {
      const res = await api.fileDiff(slug, threadId, path);
      setDiffText(res.diff);
    } catch (err) {
      setDiffText(err instanceof Error ? err.message : String(err));
    } finally {
      setDiffLoading(false);
    }
  };

  const applyUndoResult = async (
    res: {
      ok: boolean;
      summary: { restored: number; deleted: number; skipped: number; failed: number };
      results?: Array<{ rel_path: string; action: string; detail: string }>;
      changes?: FileChangesSummary;
      plan_changes?: PlanFileChangesSummary;
    },
    scope: string,
  ) => {
    if (res.changes) {
      setAgentChanges(res.changes);
    }
    if (res.plan_changes) {
      setPlanChanges(res.plan_changes);
    }
    const { restored, deleted, skipped, failed } = res.summary;
    const effective = restored + deleted;
    if (failed > 0 || skipped > 0 || effective === 0) {
      const lines =
        res.results?.map((item) => `${item.rel_path}: ${item.detail || item.action}`) ?? [];
      await alert(
        effective === 0 && failed === 0 && skipped === 0
          ? [
              `${scope}回滚未生效：没有文件被还原或删除。`,
              '账本路径与磁盘不一致时会出现此情况（例如 Agent 用 shell 写到了别的路径）。',
              ...lines,
            ].join('\n')
          : [
              `${scope}回滚未完成（成功 ${effective}，跳过 ${skipped}，失败 ${failed}）。`,
              ...lines,
            ].join('\n'),
      );
      return;
    }
    if (restored + deleted > 0) {
      const parts: string[] = [];
      if (deleted > 0) {
        parts.push(`已删除 ${deleted} 个新建文件`);
      }
      if (restored > 0) {
        parts.push(`已还原 ${restored} 个文件`);
      }
      setReviewHint(parts.join('；'));
    }
  };

  const handleUndoAll = async () => {
    if (!slug || !canUndo) {
      return;
    }
    const scope =
      mode === 'plan'
        ? '整个 Plan 的全部 Work 产出'
        : mode === 'worker' && taskId
          ? `Work ${taskId}`
          : '本会话';
    if (
      !(await confirm({
        title: '回滚文件改动',
        message: `确定回滚 ${scope} 的所有文件改动？\n将还原快照或删除本会话新建的文件。`,
        confirmLabel: '回滚',
        danger: true,
      }))
    ) {
      return;
    }
    setActionBusy(true);
    try {
      if (mode === 'plan' && planThreadId) {
        const res = await api.planUndo(slug, planThreadId, 'all');
        await applyUndoResult({ ...res, changes: undefined }, scope);
        if (res.plan_changes) {
          setPlanChanges(res.plan_changes);
        }
      } else if (mode === 'worker' && planThreadId && taskId) {
        const res = await api.planUndo(slug, planThreadId, 'all', taskId);
        await applyUndoResult({ ...res, changes: undefined }, scope);
        if (res.plan_changes) {
          setPlanChanges(res.plan_changes);
        }
      } else {
        const res = await api.undoFiles(slug, sessionThreadId, 'all');
        await applyUndoResult(res, scope);
      }
      setSelectedPath(null);
      setDiffText('');
      await refresh();
      onChangesUpdated?.();
    } catch (err) {
      await alert(err instanceof Error ? err.message : String(err));
    } finally {
      setActionBusy(false);
    }
  };

  const handleUndoOne = async (path: string, threadId: string) => {
    if (!slug) {
      return;
    }
    if (!(await confirm({ message: `回滚文件 ${path}？`, confirmLabel: '回滚', danger: true }))) {
      return;
    }
    setActionBusy(true);
    try {
      if (mode === 'plan' && planThreadId) {
        const item = flatItems.find((p) => p.path === path);
        if (item?.task_id) {
          const res = await api.planUndo(slug, planThreadId, path, item.task_id);
          await applyUndoResult({ ...res, changes: undefined }, path);
          if (res.plan_changes) {
            setPlanChanges(res.plan_changes);
          }
        }
      } else if (mode === 'worker' && planThreadId && taskId) {
        const res = await api.planUndo(slug, planThreadId, path, taskId);
        await applyUndoResult({ ...res, changes: undefined }, path);
        if (res.plan_changes) {
          setPlanChanges(res.plan_changes);
        }
      } else {
        const res = await api.undoFiles(slug, threadId, path);
        await applyUndoResult(res, path);
      }
      if (selectedPath === path) {
        setSelectedPath(null);
        setDiffText('');
      }
      await refresh();
      onChangesUpdated?.();
    } catch (err) {
      await alert(err instanceof Error ? err.message : String(err));
    } finally {
      setActionBusy(false);
    }
  };

  const handleReview = async () => {
    if (!slug || totalCount === 0) {
      return;
    }
    setActionBusy(true);
    setReviewHint('');
    try {
      if (mode === 'plan' && planThreadId) {
        const res = await api.planReview(slug, planThreadId, '');
        if (res.ok && res.reviews?.length) {
          setReviewHint(
            res.reviews.map((r) => `${r.task_id || ''}: ${r.review_path || ''}`).join('\n'),
          );
        } else {
          setReviewHint(res.message || '无变更可评审');
        }
      } else {
        const res = await api.reviewSession(slug, sessionThreadId, '');
        if (res.ok) {
          setReviewHint(`评审已落盘：${res.review_path || ''}`);
        } else {
          setReviewHint(res.message || '无变更可评审');
        }
      }
    } catch (err) {
      setReviewHint(err instanceof Error ? err.message : String(err));
    } finally {
      setActionBusy(false);
    }
  };

  const showBar = embedded ? true : totalCount > 0 || expanded;

  if (!showBar) {
    return (
      <div className="file-changes-panel file-changes-panel--empty">
        <div className="file-changes-bar">
          <span className="file-changes-count file-changes-count-muted">暂无文件改动</span>
          <span className="file-changes-meta">Worker 写文件后将在此显示 Undo / Review</span>
        </div>
      </div>
    );
  }

  const disabled = busy || actionBusy;
  const panelClass = [
    'file-changes-panel',
    embedded ? 'file-changes-panel--embedded' : '',
    expanded ? 'is-expanded' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={panelClass}>
      <div className="file-changes-bar">
        <button
          type="button"
          className="file-changes-toggle"
          onClick={() => totalCount > 0 && setExpanded((v) => !v)}
          aria-expanded={expanded}
          disabled={totalCount === 0}
        >
          <span className="file-changes-chevron">{expanded ? '▾' : '▸'}</span>
          <span className="file-changes-count">
            {totalCount > 0 ? `${totalCount} Files` : '0 Files'}
          </span>
          {mode === 'plan' && planChanges && planChanges.groups.length > 1 && (
            <span className="file-changes-meta">· {planChanges.groups.length} Works</span>
          )}
        </button>
        <div className="file-changes-actions">
          {embedded && busy && onStop && (
            <button
              type="button"
              className="file-changes-stop"
              onClick={onStop}
              title="停止生成 (Ctrl+C)"
            >
              Stop <kbd className="file-changes-kbd">^C</kbd>
            </button>
          )}
          {!embedded && (
            <button
              type="button"
              className="file-changes-btn"
              disabled={disabled || !canUndo}
              onClick={handleUndoAll}
              title="还原全部文件改动"
            >
              全部回滚
            </button>
          )}
          <button
            type="button"
            className={`file-changes-btn file-changes-btn-review${embedded ? ' is-embedded' : ' file-changes-btn-primary'}`}
            disabled={disabled || totalCount === 0}
            onClick={handleReview}
            title="LLM 代码评审（等同 /review）"
          >
            Review
          </button>
          {embedded && canUndo && totalCount > 0 && (
            <button
              type="button"
              className="file-changes-btn file-changes-btn-ghost"
              disabled={disabled}
              onClick={handleUndoAll}
              title="还原全部文件改动"
            >
              全部回滚
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <div className="file-changes-body">
          {mode === 'plan' && planChanges ? (
            planChanges.groups.map((group) => (
              <div key={group.task_id} className="file-changes-group">
                <div className="file-changes-group-head">
                  <span className="file-changes-group-id">{group.task_id}</span>
                  <span className="file-changes-group-title">{group.title}</span>
                  <button
                    type="button"
                    className="file-changes-group-undo"
                    disabled={disabled}
                    onClick={async () => {
                      if (!slug || !planThreadId) {
                        return;
                      }
                      if (
                        !(await confirm({
                          message: `回滚 Work ${group.task_id} 的全部文件改动？`,
                          confirmLabel: '回滚',
                          danger: true,
                        }))
                      ) {
                        return;
                      }
                      setActionBusy(true);
                      try {
                        await api.planUndo(slug, planThreadId, 'all', group.task_id);
                        await refresh();
                        onChangesUpdated?.();
                      } catch (err) {
                        await alert(err instanceof Error ? err.message : String(err));
                      } finally {
                        setActionBusy(false);
                      }
                    }}
                  >
                    回滚此 Work
                  </button>
                </div>
                <ul className="file-changes-list">
                  {group.paths.map((item) => (
                    <li key={`${group.task_id}:${item.path}`} className="file-changes-item">
                      <button
                        type="button"
                        className={`file-changes-path${selectedPath === item.path ? ' is-active' : ''}`}
                        onClick={() => loadDiff(item.path, group.thread_id)}
                      >
                        <span className={`file-changes-kind kind-${item.kind}`}>
                          {item.kind === 'created' ? '新建' : '修改'}
                        </span>
                        <span className="file-changes-path-text">{item.path}</span>
                      </button>
                      <button
                        type="button"
                        className="file-changes-undo-one"
                        disabled={disabled}
                        onClick={() => handleUndoOne(item.path, group.thread_id)}
                      >
                        回滚
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))
          ) : (
            <ul className="file-changes-list">
              {flatItems.map((item) => (
                <li key={item.path} className="file-changes-item">
                  <button
                    type="button"
                    className={`file-changes-path${selectedPath === item.path ? ' is-active' : ''}`}
                    onClick={() => loadDiff(item.path, sessionThreadId)}
                  >
                    <span className={`file-changes-kind kind-${item.kind}`}>
                      {item.kind === 'created' ? '新建' : '修改'}
                    </span>
                    <span className="file-changes-path-text">{item.path}</span>
                  </button>
                  <button
                    type="button"
                    className="file-changes-undo-one"
                    disabled={disabled}
                    onClick={() => handleUndoOne(item.path, sessionThreadId)}
                  >
                    回滚
                  </button>
                </li>
              ))}
            </ul>
          )}

          {selectedPath && (
            <div className="file-changes-diff">
              <div className="file-changes-diff-head">{selectedPath}</div>
              {diffLoading ? (
                <div className="file-changes-diff-loading">加载 diff…</div>
              ) : (
                <pre className="file-changes-diff-pre">{diffText}</pre>
              )}
            </div>
          )}

          {reviewHint && <div className="file-changes-review-hint">{reviewHint}</div>}
        </div>
      )}
    </div>
  );
}