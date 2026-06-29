import { useCallback, useEffect, useState } from 'react';
import { api, type Capabilities, type ContextUsage, type IndexStatus, type PlanDetail } from '../../api/client';
import type { TraceStep } from '../../types/trace';
import { partitionTraceMiscLines, filterTraceMiscWhenSteps, traceStepsFingerprint } from '../../types/trace';
import { useStickToBottomScroll } from '../../utils/useStickToBottomScroll';
import WorkflowGraph from '../WorkflowGraph';
import TraceStepList from './TraceStepList';
import TraceTurnList from './TraceTurnList';
import type { TraceTurn } from '../../types/trace';
import { useAppDialog } from '../AppDialog';
import {
  CONTEXT_BREAKDOWN_LABELS,
  formatContextTokens,
} from '../../utils/contextDisplay';

interface TraceLine {
  id: string;
  text: string;
}

export type RightPanelTab = 'trace' | 'tools' | 'context' | 'log' | 'plan';

interface Props {
  caps: Capabilities | null;
  traceLines: TraceLine[];
  traceSteps: TraceStep[];
  traceTurns?: TraceTurn[];
  liveThinking?: string;
  planDetail: PlanDetail | null;
  slug: string;
  threadId: string;
  isPlan: boolean;
  isAgent?: boolean;
  allowWrite?: boolean;
  requestedTab?: RightPanelTab | null;
  onRequestedTabHandled?: () => void;
  onTraceMode: (mode: string) => void;
  onPlanConfirm: () => void;
  onPlanContinue: () => void;
  busy: boolean;
  contextRefreshSignal?: number;
  onTaskSelect?: (taskId: string) => void;
  onCapsLoaded?: (caps: Capabilities) => void;
  /** 当前轮已执行秒数（SSE 心跳 + 本地计时） */
  traceActivitySec?: number;
}

function miscLineClass(line: string): string {
  if (line.includes('思考中')) {
    return 'cursor-trace-misc cursor-trace-misc--thinking';
  }
  if (line.includes('用户消息')) {
    return 'cursor-trace-misc cursor-trace-misc--user';
  }
  if (line.includes('并行执行') || line.startsWith('▶')) {
    return 'cursor-trace-misc cursor-trace-misc--milestone';
  }
  if (line.includes('提示:') || line.includes('实时:')) {
    return 'cursor-trace-misc cursor-trace-misc--hint';
  }
  if (line.includes('本轮完成')) {
    return 'cursor-trace-misc cursor-trace-misc--milestone';
  }
  return 'cursor-trace-misc';
}

function traceModeHint(mode: string): string {
  switch (mode) {
    case 'all':
      return '完整过程：步骤详情 + 原始日志行';
    case 'steps':
      return '折叠步骤摘要（默认）';
    case 'reply':
      return '仅助手回复；过程 trace 不展示';
    case 'none':
      return '静默：不展示过程 trace';
    default:
      return '';
  }
}

export default function CursorRightPanel({
  caps,
  traceLines,
  traceSteps,
  traceTurns = [],
  liveThinking = '',
  planDetail,
  slug,
  threadId,
  isPlan,
  isAgent = false,
  allowWrite = false,
  requestedTab = null,
  onRequestedTabHandled,
  onTraceMode,
  onPlanConfirm,
  onPlanContinue,
  busy,
  contextRefreshSignal = 0,
  onTaskSelect,
  onCapsLoaded,
  traceActivitySec = 0,
}: Props) {
  const { confirm } = useAppDialog();
  const [tab, setTab] = useState<RightPanelTab>(isPlan ? 'plan' : 'trace');
  const [toolsCaps, setToolsCaps] = useState<Capabilities | null>(caps);
  const [toolsLoading, setToolsLoading] = useState(false);
  const [toolsError, setToolsError] = useState('');
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [contextBusy, setContextBusy] = useState(false);
  const [indexBusy, setIndexBusy] = useState(false);
  const [contextMsg, setContextMsg] = useState('');
  const [logLines, setLogLines] = useState<string[]>([]);
  const [logBusy, setLogBusy] = useState(false);

  useEffect(() => {
    if (caps) {
      setToolsCaps(caps);
    }
  }, [caps]);

  useEffect(() => {
    setTab(isPlan ? 'plan' : 'trace');
  }, [isPlan, threadId]);

  const loadToolsCaps = useCallback(async () => {
    if (!slug) {
      return;
    }
    setToolsLoading(true);
    setToolsError('');
    try {
      const data = await api.capabilities(slug, allowWrite);
      setToolsCaps(data);
      onCapsLoaded?.(data);
    } catch (err) {
      setToolsError(err instanceof Error ? err.message : String(err));
    } finally {
      setToolsLoading(false);
    }
  }, [slug, allowWrite, onCapsLoaded]);

  useEffect(() => {
    if (tab !== 'tools' || !slug) {
      return;
    }
    void loadToolsCaps();
  }, [tab, slug, allowWrite, loadToolsCaps]);

  const refreshContextPanel = useCallback(async () => {
    if (!slug) {
      return;
    }
    setContextBusy(true);
    try {
      const [ctx, idx] = await Promise.all([
        api.contextUsage(slug, allowWrite, isAgent ? threadId : ''),
        api.indexStatus(slug),
      ]);
      setContextUsage(ctx);
      setIndexStatus(idx);
    } catch (err) {
      setContextMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setContextBusy(false);
    }
  }, [slug, allowWrite, isAgent, threadId]);

  const refreshLogPanel = useCallback(async () => {
    if (!slug) {
      return;
    }
    setLogBusy(true);
    try {
      const res = await api.executionLog(slug, 40);
      setLogLines(res.lines);
    } catch (err) {
      setLogLines([err instanceof Error ? err.message : String(err)]);
    } finally {
      setLogBusy(false);
    }
  }, [slug]);

  useEffect(() => {
    if (tab === 'context') {
      void refreshContextPanel();
    }
    if (tab === 'log') {
      void refreshLogPanel();
    }
  }, [tab, refreshContextPanel, refreshLogPanel]);

  useEffect(() => {
    if (tab === 'context') {
      void refreshContextPanel();
    }
  }, [contextRefreshSignal, tab, refreshContextPanel]);

  useEffect(() => {
    if (!busy && tab === 'context') {
      void refreshContextPanel();
    }
  }, [busy, tab, refreshContextPanel]);

  useEffect(() => {
    if (requestedTab) {
      setTab(requestedTab);
      onRequestedTabHandled?.();
    }
  }, [requestedTab, onRequestedTabHandled]);

  const traceMode = caps?.trace_mode ?? 'steps';
  const showProcess = traceMode === 'steps' || traceMode === 'all';
  const fullLog = traceLines.map((l) => l.text).join('\n');
  const miscLines = partitionTraceMiscLines(fullLog.split('\n'));
  const displayMiscLines = filterTraceMiscWhenSteps(miscLines, traceSteps.length);
  const hasStructuredSteps = traceSteps.length > 0 || traceTurns.some((turn) => turn.steps.length > 0);
  const hasLogLines = fullLog.trim().length > 0;
  const thinkingChars = liveThinking.trim().length;
  /** all：完整日志；steps：里程碑/用户消息 + 结构化步骤（与终端一致） */
  const showFullLog = traceMode === 'all' && hasLogLines;
  const showMiscLines = showProcess && traceMode === 'steps' && displayMiscLines.length > 0;
  const hasTrace =
    (showProcess && (hasLogLines || hasStructuredSteps)) || thinkingChars > 0;
  const modeHint = traceModeHint(traceMode);

  const traceScroll = useStickToBottomScroll<HTMLDivElement>(
    [
      traceStepsFingerprint(traceSteps),
      traceTurns.map((t) => `${t.id}:${t.steps.length}`).join('|'),
      traceLines.length,
      traceLines.map((l) => l.text).join('\n').length,
      liveThinking.length,
      busy,
      traceMode,
      showFullLog,
      displayMiscLines.length,
      traceActivitySec,
    ],
    { enabled: tab === 'trace', resetKey: threadId },
  );

  return (
    <aside className="cursor-right">
      <div className="cursor-right-tabs">
        {isPlan && (
          <button type="button" className={tab === 'plan' ? 'active' : ''} onClick={() => setTab('plan')}>
            Plan
          </button>
        )}
        <button type="button" className={tab === 'trace' ? 'active' : ''} onClick={() => setTab('trace')}>
          Trace
        </button>
        <button type="button" className={tab === 'tools' ? 'active' : ''} onClick={() => setTab('tools')}>
          Tools
        </button>
        <button type="button" className={tab === 'context' ? 'active' : ''} onClick={() => setTab('context')}>
          Context
        </button>
        <button type="button" className={tab === 'log' ? 'active' : ''} onClick={() => setTab('log')}>
          Log
        </button>
      </div>

      <div className="cursor-right-body">
        {tab === 'trace' && (
          <div className="cursor-trace-panel">
            {caps && (
              <div className="cursor-trace-modes">
                {['all', 'steps', 'reply', 'none'].map((m) => (
                  <button
                    key={m}
                    type="button"
                    className={traceMode === m ? 'chip active' : 'chip'}
                    onClick={() => onTraceMode(m)}
                  >
                    {m}
                  </button>
                ))}
              </div>
            )}
            {busy && traceMode === 'none' && !thinkingChars && (
              <p className="cursor-trace-mode-hint cursor-trace-mode-hint--active">
                执行中：当前为 none，过程不展示。可切到{' '}
                <button type="button" className="cursor-link-btn" onClick={() => onTraceMode('steps')}>
                  steps
                </button>{' '}
                查看。
              </p>
            )}
            {modeHint && <p className="cursor-trace-mode-hint">{modeHint}</p>}
            {busy && traceActivitySec >= 5 && showProcess && (
              <p className="cursor-trace-mode-hint cursor-trace-mode-hint--active">
                ⏳ 仍在执行… {traceActivitySec}s（工具或 LLM 阻塞时可能数十秒无新步骤，属正常）
              </p>
            )}
            <div ref={traceScroll.ref} className="cursor-trace-log">
              {!showProcess && !thinkingChars && (
                <div className="muted small">
                  当前模式不展示过程 trace。切换为 steps 或 all 可查看步骤记录。
                </div>
              )}
              {thinkingChars > 0 && (
                <details className="cursor-trace-thinking" open={busy}>
                  <summary className="cursor-trace-thinking-summary">
                    ◎ 模型思考（{thinkingChars} 字{busy ? ' · 流式' : ''}）
                  </summary>
                  <pre className="cursor-trace-thinking-body">
                    {thinkingChars > 12000
                      ? liveThinking.trim().slice(-12000)
                      : liveThinking.trim()}
                  </pre>
                </details>
              )}
              {showProcess && !hasTrace && (
                <div className="muted small">
                  {busy
                    ? '正在执行，过程将实时显示…'
                    : '暂无 trace。仅在 Web 上正常结束的回合会落盘；中断、无限循环或纯终端会话通常无记录。'}
                </div>
              )}
              {showProcess && showFullLog && (
                <pre className="cursor-trace-full-log">{fullLog}</pre>
              )}
              {showMiscLines && (
                <div className="cursor-trace-misc-block">
                  {displayMiscLines.map((line, index) => (
                    <div key={index} className={miscLineClass(line)}>
                      {line}
                    </div>
                  ))}
                </div>
              )}
              {showProcess && (traceTurns.length > 0 || hasStructuredSteps) && (
                <div className="cursor-trace-steps">
                  {traceTurns.length > 0 ? (
                    <TraceTurnList
                      turns={traceTurns}
                      defaultOpenLast={busy}
                      expandBodies={traceMode === 'all'}
                    />
                  ) : (
                    <>
                      <div className="cursor-trace-steps-label">步骤（可展开详情）</div>
                      <TraceStepList
                        steps={traceSteps}
                        defaultOpenLast={busy}
                        expandBodies={traceMode === 'all'}
                      />
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {tab === 'tools' && (
          <div className="cursor-tools-panel">
            {toolsLoading && !toolsCaps && (
              <p className="muted small">加载工具列表…</p>
            )}
            {toolsError && !toolsCaps && (
              <div className="cursor-tools-error">
                <p className="small">{toolsError}</p>
                <button type="button" className="cursor-link-btn" onClick={() => void loadToolsCaps()}>
                  重试
                </button>
              </div>
            )}
            {toolsCaps && (
              <>
            <section>
              <h4>内置 ({toolsCaps.builtin_tools.length})</h4>
              <ul className="cursor-catalog-list">
                {toolsCaps.builtin_tools.map((t) => (
                  <li key={t.name} className="cursor-catalog-item">
                    <div className="cursor-catalog-item-name">{t.name}</div>
                    {t.description?.trim() && (
                      <p
                        className="cursor-catalog-item-desc cursor-catalog-item-desc--clamp"
                        title={t.description.trim()}
                      >
                        {t.description.trim()}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </section>
            <section>
              <h4>MCP ({toolsCaps.mcp_tools.length})</h4>
              <p className="small muted">{toolsCaps.mcp_summary}</p>
              <ul className="cursor-catalog-list">
                {toolsCaps.mcp_tools.map((t) => (
                  <li key={t.name} className="cursor-catalog-item">
                    <div className="cursor-catalog-item-name">{t.name}</div>
                    {t.description?.trim() && (
                      <p
                        className="cursor-catalog-item-desc cursor-catalog-item-desc--clamp"
                        title={t.description.trim()}
                      >
                        {t.description.trim()}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </section>
            <section>
              <h4>Skills ({toolsCaps.skills.length})</h4>
              <ul className="cursor-catalog-list">
                {toolsCaps.skills.map((s) => (
                  <li key={s.name} className="cursor-catalog-item">
                    <div className="cursor-catalog-item-name">
                      {s.name}
                      <span className="muted"> · {s.scope_label || (s.scope === 'user' ? '个人' : '项目')}</span>
                    </div>
                    {s.description?.trim() && (
                      <p
                        className="cursor-catalog-item-desc cursor-catalog-item-desc--clamp"
                        title={s.description.trim()}
                      >
                        {s.description.trim()}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </section>
              </>
            )}
            {!toolsLoading && toolsCaps && toolsCaps.builtin_tools.length === 0 && toolsCaps.mcp_tools.length === 0 && toolsCaps.skills.length === 0 && (
              <p className="muted small">当前工作区暂无可用工具</p>
            )}
          </div>
        )}

        {tab === 'context' && (
          <div className="cursor-context-panel">
            <section className="cursor-context-section">
              <div className="cursor-context-section-head">
                <h4>上下文用量</h4>
                <button
                  type="button"
                  className="cursor-link-btn"
                  disabled={contextBusy}
                  onClick={() => void refreshContextPanel()}
                >
                  刷新
                </button>
              </div>
              {contextBusy && !contextUsage && <p className="muted small">加载中…</p>}
              {contextUsage && (
                <>
                  <div className="cursor-context-meter">
                    <div className="cursor-context-meter-label">
                      <span className={contextUsage.pct >= 85 ? 'cursor-context-warn' : ''}>
                        {contextUsage.pct}% Full
                      </span>
                      <span>
                        {formatContextTokens(contextUsage.total)} / {formatContextTokens(contextUsage.limit)} Tokens
                      </span>
                    </div>
                    <div className="cursor-context-bar">
                      <div
                        className="cursor-context-bar-fill"
                        style={{ width: `${Math.min(100, contextUsage.pct)}%` }}
                      />
                    </div>
                  </div>
                  <ul className="cursor-context-breakdown">
                    {Object.entries(contextUsage.breakdown)
                      .filter(([, v]) => v > 0)
                      .map(([key, value]) => (
                        <li key={key}>
                          <span>{CONTEXT_BREAKDOWN_LABELS[key] || key}</span>
                          <span>{formatContextTokens(value)}</span>
                        </li>
                      ))}
                  </ul>
                  <p className="small muted">
                    消息 {contextUsage.message_count} · 工具 {contextUsage.tool_count}
                    {contextUsage.mcp_tool_count > 0 ? `（MCP ${contextUsage.mcp_tool_count}）` : ''}
                  </p>
                  {contextUsage.budget_note && (
                    <p className="small muted">{contextUsage.budget_note}</p>
                  )}
                  {isAgent && contextUsage.has_session && (
                    <button
                      type="button"
                      className="cursor-btn-ghost"
                      disabled={busy || contextBusy}
                      onClick={async () => {
                        setContextBusy(true);
                        setContextMsg('');
                        try {
                          const res = await api.compressContext(slug, threadId, allowWrite);
                          setContextMsg(res.message);
                          await refreshContextPanel();
                        } catch (err) {
                          setContextMsg(err instanceof Error ? err.message : String(err));
                        } finally {
                          setContextBusy(false);
                        }
                      }}
                    >
                      压缩历史
                    </button>
                  )}
                  {isAgent && !contextUsage.has_session && (
                    <p className="small muted">选择 Agent 会话后可压缩历史</p>
                  )}
                </>
              )}
            </section>

            <section className="cursor-context-section">
              <div className="cursor-context-section-head">
                <h4>代码索引</h4>
                <button
                  type="button"
                  className="cursor-link-btn"
                  disabled={indexBusy}
                  onClick={() => void refreshContextPanel()}
                >
                  刷新
                </button>
              </div>
              {indexStatus && (
                <>
                  <ul className="cursor-index-stats">
                    <li>
                      <span>状态</span>
                      <span>{indexStatus.exists ? '已建立' : '未建立'}</span>
                    </li>
                    <li>
                      <span>Chunks</span>
                      <span>{indexStatus.chunk_count}</span>
                    </li>
                    <li>
                      <span>向量维度</span>
                      <span>{indexStatus.vector_dim || '—'}</span>
                    </li>
                    <li>
                      <span>Manifest</span>
                      <span>{indexStatus.manifest_files} 文件</span>
                    </li>
                    <li>
                      <span>最近索引</span>
                      <span>{indexStatus.last_indexed_at || '无'}</span>
                    </li>
                    <li>
                      <span>Embedding</span>
                      <span className="cursor-index-embed">{indexStatus.embedding}</span>
                    </li>
                    <li>
                      <span>Watch</span>
                      <span>
                        {indexStatus.watch_enabled
                          ? indexStatus.watch_with_agent
                            ? '配置开启（Web 未监听）'
                            : '配置开启'
                          : '关闭'}
                      </span>
                    </li>
                  </ul>
                  <div className="cursor-index-actions">
                    {(['incremental', 'full', 'rebuild'] as const).map((action) => (
                      <button
                        key={action}
                        type="button"
                        className="cursor-btn-ghost"
                        disabled={indexBusy}
                        onClick={async () => {
                          setIndexBusy(true);
                          setContextMsg('');
                          try {
                            const res = await api.runIndex(slug, action);
                            setContextMsg(
                              res.ok
                                ? `${action} 完成${res.log_path ? ` · 日志 ${res.log_path}` : ''}`
                                : `${action} 失败（exit ${res.exit_code}）`,
                            );
                            await refreshContextPanel();
                          } catch (err) {
                            setContextMsg(err instanceof Error ? err.message : String(err));
                          } finally {
                            setIndexBusy(false);
                          }
                        }}
                      >
                        {action === 'incremental' ? '增量' : action === 'full' ? '全量' : '重建'}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </section>

            {contextMsg && <pre className="cursor-context-msg">{contextMsg}</pre>}
          </div>
        )}

        {tab === 'log' && (
          <div className="cursor-log-panel">
            <div className="cursor-context-section-head">
              <h4>执行日志</h4>
              <button
                type="button"
                className="cursor-link-btn"
                disabled={logBusy}
                onClick={() => void refreshLogPanel()}
              >
                刷新
              </button>
            </div>
            {logBusy && <p className="muted small">加载中…</p>}
            {!logBusy && logLines.length === 0 && (
              <p className="muted small">暂无执行日志</p>
            )}
            <pre className="cursor-log-lines">
              {logLines.join('\n') || '（空）'}
            </pre>
            <button
              type="button"
              className="cursor-btn-ghost"
              disabled={logBusy}
              onClick={async () => {
                if (!slug || !(await confirm({ message: '清理过期日志？', confirmLabel: '清理' }))) {
                  return;
                }
                setLogBusy(true);
                try {
                  const res = await api.purgeExecutionLog(slug);
                  setLogLines([res.message]);
                } catch (err) {
                  setLogLines([err instanceof Error ? err.message : String(err)]);
                } finally {
                  setLogBusy(false);
                }
              }}
            >
              清理过期日志
            </button>
          </div>
        )}


        {tab === 'plan' && planDetail && (
          <div className="cursor-plan-panel">
            <div className="cursor-plan-meta">
              <span className="badge">{planDetail.phase}</span>
              {planDetail.job?.running && <span className="badge badge-running">running</span>}
            </div>
            <p className="small">{planDetail.goal}</p>
            <WorkflowGraph
              slug={slug}
              threadId={threadId}
              nodes={planDetail.workflow_snapshot?.nodes || []}
              tasks={planDetail.workflow_snapshot?.tasks || []}
              planTasks={planDetail.tasks}
              synthesizeDependsOn={planDetail.workflow_snapshot?.synthesize_depends_on}
              currentTaskId={
                (planDetail.workflow_snapshot as { current_task_id?: string })?.current_task_id || null
              }
              variant="main"
              onTaskSelect={onTaskSelect}
            />
            <div className="cursor-plan-actions">
              {planDetail.phase === 'awaiting_confirm' && (
                <button type="button" className="cursor-btn-primary" onClick={onPlanConfirm}>
                  确认计划
                </button>
              )}
              {(() => {
                const wfTasks = planDetail.workflow_snapshot?.tasks || [];
                const planTasks = planDetail.tasks || [];
                const taskIds =
                  planTasks.length > 0
                    ? planTasks.map((t) => String(t.id || ''))
                    : wfTasks.map((t) => t.id);
                const statusOf = (taskId: string): string => {
                  const wf = wfTasks.find((t) => t.id === taskId);
                  const row = planTasks.find((t) => String(t.id) === taskId);
                  return String(wf?.worker_node_status || wf?.status || row?.status || 'pending');
                };
                const incompleteCount = taskIds.filter((id) => {
                  const s = statusOf(id);
                  return s === 'pending' || s === 'running' || s === 'failed';
                }).length;
                const showContinue =
                  planDetail.phase === 'executing' &&
                  incompleteCount > 0 &&
                  !planDetail.job?.running;
                const showSynthesizeContinue =
                  incompleteCount === 0 &&
                  !planDetail.final_report &&
                  !planDetail.job?.running &&
                  planDetail.phase === 'executing';
                if (planDetail.phase !== 'executing') {
                  return null;
                }
                if (showContinue) {
                  return (
                    <button
                      type="button"
                      className="cursor-btn-primary"
                      onClick={onPlanContinue}
                      disabled={busy}
                    >
                      继续执行未完成（已成功跳过）
                    </button>
                  );
                }
                if (showSynthesizeContinue) {
                  return (
                    <button
                      type="button"
                      className="cursor-btn-ghost"
                      onClick={onPlanContinue}
                      disabled={busy || !!planDetail.job?.running}
                    >
                      Continue
                    </button>
                  );
                }
                return null;
              })()}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
