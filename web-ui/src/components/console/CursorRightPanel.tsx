import { useCallback, useEffect, useState } from 'react';
import { api, type Capabilities, type ContextUsage, type IndexStatus, type PlanDetail } from '../../api/client';
import type { TraceStep } from '../../types/trace';
import { partitionTraceMiscLines } from '../../types/trace';
import WorkflowGraph from '../WorkflowGraph';
import TraceStepList from './TraceStepList';

interface TraceLine {
  id: string;
  text: string;
}

interface Props {
  caps: Capabilities | null;
  traceLines: TraceLine[];
  traceSteps: TraceStep[];
  liveThinking?: string;
  planDetail: PlanDetail | null;
  slug: string;
  threadId: string;
  isPlan: boolean;
  isAgent?: boolean;
  allowWrite?: boolean;
  onMetaOutput?: (text: string) => void;
  onTraceMode: (mode: string) => void;
  onPlanConfirm: () => void;
  onPlanContinue: () => void;
  busy: boolean;
  onTaskSelect?: (taskId: string) => void;
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

function formatTokens(n: number): string {
  if (n >= 10_000) {
    return `~${(n / 1000).toFixed(1)}K`;
  }
  if (n >= 1000) {
    return `~${(n / 1000).toFixed(1)}K`;
  }
  return `~${n}`;
}

const BREAKDOWN_LABELS: Record<string, string> = {
  system_prompt: 'System prompt',
  tool_definitions: 'Tool definitions',
  rules: 'Rules',
  skills: 'Skills',
  mcp: 'MCP',
  markdowns_index: 'Markdowns index',
  summarized_conversation: 'Summarized conversation',
  conversation: 'Conversation',
};

export default function CursorRightPanel({
  caps,
  traceLines,
  traceSteps,
  liveThinking = '',
  planDetail,
  slug,
  threadId,
  isPlan,
  isAgent = false,
  allowWrite = false,
  onMetaOutput,
  onTraceMode,
  onPlanConfirm,
  onPlanContinue,
  busy,
  onTaskSelect,
}: Props) {
  const [tab, setTab] = useState<'trace' | 'tools' | 'context' | 'log' | 'quick' | 'plan'>(
    isPlan ? 'plan' : 'trace',
  );
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [contextBusy, setContextBusy] = useState(false);
  const [indexBusy, setIndexBusy] = useState(false);
  const [contextMsg, setContextMsg] = useState('');
  const [logLines, setLogLines] = useState<string[]>([]);
  const [logBusy, setLogBusy] = useState(false);
  const [metaBusy, setMetaBusy] = useState(false);
  const [metaMsg, setMetaMsg] = useState('');

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

  const runMeta = async (command: string) => {
    if (!slug) {
      return;
    }
    setMetaBusy(true);
    setMetaMsg('');
    try {
      const res = await api.metaCommand(slug, command, allowWrite, isAgent ? threadId : '');
      const out = res.output?.trim() || '（无输出）';
      setMetaMsg(out);
      onMetaOutput?.(out);
    } catch (err) {
      setMetaMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setMetaBusy(false);
    }
  };

  const traceMode = caps?.trace_mode ?? 'steps';
  const sandbox = caps?.sandbox;
  const showProcess = traceMode === 'steps' || traceMode === 'all';
  const fullLog = traceLines.map((l) => l.text).join('\n');
  const miscLines = partitionTraceMiscLines(fullLog.split('\n'));
  const hasStructuredSteps = traceSteps.length > 0;
  const hasLogLines = fullLog.trim().length > 0;
  const thinkingChars = liveThinking.trim().length;
  /** all：完整日志；steps：里程碑/用户消息 + 结构化步骤（与终端一致） */
  const showFullLog = traceMode === 'all' && hasLogLines;
  const showMiscLines = showProcess && traceMode === 'steps' && miscLines.length > 0;
  const hasTrace =
    (showProcess && (hasLogLines || hasStructuredSteps)) || thinkingChars > 0;
  const modeHint = traceModeHint(traceMode);

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
        <button type="button" className={tab === 'quick' ? 'active' : ''} onClick={() => setTab('quick')}>
          快捷
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
            <div className="cursor-trace-log">
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
                  {miscLines.map((line, index) => (
                    <div key={index} className={miscLineClass(line)}>
                      {line}
                    </div>
                  ))}
                </div>
              )}
              {showProcess && hasStructuredSteps && (
                <div className="cursor-trace-steps">
                  <div className="cursor-trace-steps-label">步骤（可展开详情）</div>
                  <TraceStepList
                    steps={traceSteps}
                    defaultOpenLast={busy}
                    expandBodies={traceMode === 'all'}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {tab === 'tools' && caps && (
          <div className="cursor-tools-panel">
            <section>
              <h4>内置 ({caps.builtin_tools.length})</h4>
              <ul className="cursor-catalog-list">
                {caps.builtin_tools.map((t) => (
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
              <h4>MCP ({caps.mcp_tools.length})</h4>
              <p className="small muted">{caps.mcp_summary}</p>
              <ul className="cursor-catalog-list">
                {caps.mcp_tools.map((t) => (
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
              <h4>Skills ({caps.skills.length})</h4>
              <ul className="cursor-catalog-list">
                {caps.skills.map((s) => (
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
                        {formatTokens(contextUsage.total)} / {formatTokens(contextUsage.limit)} Tokens
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
                          <span>{BREAKDOWN_LABELS[key] || key}</span>
                          <span>{formatTokens(value)}</span>
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
                if (!slug || !window.confirm('清理过期日志？')) {
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

        {tab === 'quick' && (
          <div className="cursor-quick-panel">
            {sandbox && (
              <section className="cursor-quick-sandbox">
                <h4>沙箱</h4>
                <p className="small">
                  {sandbox.enabled
                    ? `已启用 · ${sandbox.backend || 'unknown'} · mode=${sandbox.mode} · net=${sandbox.network}`
                    : sandbox.active
                      ? `已配置但未就绪（backend=${sandbox.backend || '无'}）`
                      : '未启用'}
                </p>
              </section>
            )}
            <section>
              <h4>元命令</h4>
              <div className="cursor-quick-actions">
                <button type="button" className="cursor-btn-ghost" disabled={metaBusy} onClick={() => void runMeta('/help')}>
                  帮助
                </button>
                <button type="button" className="cursor-btn-ghost" disabled={metaBusy} onClick={() => void runMeta('/sessions')}>
                  会话列表
                </button>
                <button type="button" className="cursor-btn-ghost" disabled={metaBusy} onClick={() => void runMeta('/context')}>
                  上下文文本
                </button>
                <button type="button" className="cursor-btn-ghost" disabled={metaBusy} onClick={() => void runMeta('/index status')}>
                  索引状态
                </button>
                <button
                  type="button"
                  className="cursor-btn-ghost"
                  disabled={metaBusy}
                  onClick={() => {
                    setTab('log');
                  }}
                >
                  执行日志
                </button>
                {isAgent && (
                  <button
                    type="button"
                    className="cursor-btn-ghost"
                    disabled={metaBusy || contextBusy}
                    onClick={async () => {
                      if (!slug || !threadId) {
                        return;
                      }
                      setMetaBusy(true);
                      try {
                        const res = await api.compressContext(slug, threadId, allowWrite);
                        setMetaMsg(res.message);
                        onMetaOutput?.(res.message);
                      } catch (err) {
                        setMetaMsg(err instanceof Error ? err.message : String(err));
                      } finally {
                        setMetaBusy(false);
                      }
                    }}
                  >
                    压缩历史
                  </button>
                )}
              </div>
            </section>
            {metaMsg && <pre className="cursor-context-msg">{metaMsg}</pre>}
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
                const incomplete = wfTasks.some(
                  (t) => t.status === 'pending' || t.status === 'running' || t.status === 'failed',
                );
                if (planDetail.phase !== 'executing') {
                  return null;
                }
                if (incomplete && !planDetail.job?.running) {
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
                if (!incomplete) {
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
