import { useCallback, useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react';
import { api, type Capabilities, type ContextUsage, type IndexStatus } from '../../api/client';
import {
  CONTEXT_BREAKDOWN_LABELS,
  formatContextTokens,
  resolveSlashMetaModalRoute,
} from '../../utils/contextDisplay';

export type OpsRequest =
  | { kind: 'context' }
  | { kind: 'index' }
  | { kind: 'meta'; command: string; title: string };

export interface ConsoleOpsHandle {
  openFromSlash: (text: string) => boolean;
  showMetaOutput: (title: string, body: string) => void;
  refreshContext: () => Promise<void>;
}

interface Props {
  slug: string;
  threadId: string;
  isAgent: boolean;
  allowWrite: boolean;
  busy?: boolean;
  /** 递增时刷新上下文占用（对话轮次结束、压缩后等） */
  contextRefreshSignal?: number;
  sandbox?: Capabilities['sandbox'];
  onOpenLog?: () => void;
  opsRequest?: OpsRequest | null;
  onOpsRequestHandled?: () => void;
}

type ModalKind = 'context' | 'index' | 'text' | null;

function sandboxLabel(sandbox: NonNullable<Capabilities['sandbox']>): string {
  if (sandbox.enabled) {
    return `已启用 · ${sandbox.backend || 'unknown'}`;
  }
  if (sandbox.active) {
    return `未就绪 · ${sandbox.backend || '无后端'}`;
  }
  return '未启用';
}

function OpsModalShell({
  title,
  onClose,
  children,
  wide,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="modal-overlay cursor-ops-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        className={`modal cursor-ops-modal${wide ? ' cursor-ops-modal--wide' : ''}`}
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="cursor-ops-modal-head">
          <h2>{title}</h2>
          <button type="button" className="cursor-ops-close" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </header>
        <div className="cursor-ops-modal-body">{children}</div>
      </div>
    </div>
  );
}

function ProgressStrip({ label }: { label: string }) {
  return (
    <div className="cursor-ops-progress">
      <div className="cursor-ops-progress-bar" />
      <p className="cursor-ops-progress-label">{label}</p>
    </div>
  );
}

const ConsoleOps = forwardRef<ConsoleOpsHandle, Props>(function ConsoleOps(
  {
    slug,
    threadId,
    isAgent,
    allowWrite,
    busy = false,
    contextRefreshSignal = 0,
    sandbox,
    onOpenLog,
    opsRequest = null,
    onOpsRequestHandled,
  },
  ref,
) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [modal, setModal] = useState<ModalKind>(null);
  const [contextPct, setContextPct] = useState<number | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);
  const [contextLoading, setContextLoading] = useState(false);
  const [compressing, setCompressing] = useState(false);
  const [contextNote, setContextNote] = useState('');

  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexLoading, setIndexLoading] = useState(false);
  const [indexRunning, setIndexRunning] = useState(false);
  const [indexNote, setIndexNote] = useState('');

  const [textTitle, setTextTitle] = useState('');
  const [textBody, setTextBody] = useState('');
  const [textLoading, setTextLoading] = useState(false);

  const closeMenu = useCallback(() => setMenuOpen(false), []);
  const closeModal = useCallback(() => setModal(null), []);

  const refreshContextPct = useCallback(async () => {
    if (!slug || !isAgent) {
      setContextPct(null);
      return;
    }
    try {
      const usage = await api.contextUsage(slug, allowWrite, threadId);
      setContextPct(usage.pct);
    } catch {
      setContextPct(null);
    }
  }, [slug, isAgent, allowWrite, threadId]);

  useEffect(() => {
    void refreshContextPct();
  }, [refreshContextPct, contextRefreshSignal]);

  useEffect(() => {
    if (!isAgent || !slug || !busy) {
      return;
    }
    void refreshContextPct();
    const id = window.setInterval(() => {
      void refreshContextPct();
    }, 8000);
    return () => window.clearInterval(id);
  }, [busy, isAgent, slug, refreshContextPct]);

  useEffect(() => {
    if (!menuOpen) {
      return;
    }
    const onDocClick = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) {
        closeMenu();
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [menuOpen, closeMenu]);

  const loadContextModal = useCallback(async () => {
    if (!slug) {
      return;
    }
    setContextLoading(true);
    setContextNote('');
    try {
      const usage = await api.contextUsage(slug, allowWrite, isAgent ? threadId : '');
      setContextUsage(usage);
      if (isAgent) {
        setContextPct(usage.pct);
      }
    } catch (err) {
      setContextUsage(null);
      setContextNote(err instanceof Error ? err.message : String(err));
    } finally {
      setContextLoading(false);
    }
  }, [slug, allowWrite, isAgent, threadId]);

  useEffect(() => {
    if (modal === 'context') {
      void loadContextModal();
    }
  }, [contextRefreshSignal, modal, loadContextModal]);

  const openContext = useCallback(() => {
    closeMenu();
    setModal('context');
    void loadContextModal();
  }, [closeMenu, loadContextModal]);

  const loadIndexModal = useCallback(async () => {
    if (!slug) {
      return;
    }
    setIndexLoading(true);
    setIndexNote('');
    try {
      setIndexStatus(await api.indexStatus(slug));
    } catch (err) {
      setIndexStatus(null);
      setIndexNote(err instanceof Error ? err.message : String(err));
    } finally {
      setIndexLoading(false);
    }
  }, [slug]);

  const openIndex = useCallback(() => {
    closeMenu();
    setModal('index');
    void loadIndexModal();
  }, [closeMenu, loadIndexModal]);

  const showMetaOutput = useCallback((title: string, body: string) => {
    closeMenu();
    setModal('text');
    setTextTitle(title);
    setTextBody(body);
    setTextLoading(false);
  }, [closeMenu]);

  const runMetaText = useCallback(
    async (command: string, title: string) => {
      if (!slug) {
        return;
      }
      closeMenu();
      setModal('text');
      setTextTitle(title);
      setTextBody('');
      setTextLoading(true);
      try {
        const res = await api.metaCommand(slug, command, allowWrite, isAgent ? threadId : '');
        setTextBody(res.output?.trim() || '（无输出）');
      } catch (err) {
        setTextBody(err instanceof Error ? err.message : String(err));
      } finally {
        setTextLoading(false);
      }
    },
    [slug, allowWrite, isAgent, threadId, closeMenu],
  );

  const applyOpsRequest = useCallback(
    (req: OpsRequest) => {
      if (req.kind === 'context') {
        openContext();
        return;
      }
      if (req.kind === 'index') {
        openIndex();
        return;
      }
      void runMetaText(req.command, req.title);
    },
    [openContext, openIndex, runMetaText],
  );

  useEffect(() => {
    if (!opsRequest) {
      return;
    }
    applyOpsRequest(opsRequest);
    onOpsRequestHandled?.();
  }, [opsRequest, applyOpsRequest, onOpsRequestHandled]);

  useImperativeHandle(
    ref,
    () => ({
      openFromSlash(text: string) {
        const matched = resolveSlashMetaModalRoute(text);
        if (!matched) {
          return false;
        }
        if (matched.kind === 'context') {
          openContext();
        } else if (matched.kind === 'index') {
          openIndex();
        } else {
          void runMetaText(matched.command, matched.title);
        }
        return true;
      },
      showMetaOutput,
      refreshContext: refreshContextPct,
    }),
    [openContext, openIndex, runMetaText, refreshContextPct, showMetaOutput],
  );

  const compressHistory = async () => {
    if (!slug || !threadId || !isAgent) {
      return;
    }
    setCompressing(true);
    setContextNote('');
    try {
      const res = await api.compressContext(slug, threadId, allowWrite);
      setContextNote(res.message);
      await loadContextModal();
    } catch (err) {
      setContextNote(err instanceof Error ? err.message : String(err));
    } finally {
      setCompressing(false);
    }
  };

  const runIndexAction = async (action: 'incremental' | 'full' | 'rebuild') => {
    if (!slug) {
      return;
    }
    setIndexRunning(true);
    setIndexNote('');
    try {
      const res = await api.runIndex(slug, action);
      setIndexNote(
        res.ok
          ? `${action === 'incremental' ? '增量' : action === 'full' ? '全量' : '重建'}完成${res.log_path ? ` · 日志 ${res.log_path}` : ''}`
          : `${action} 失败（exit ${res.exit_code}）`,
      );
      await loadIndexModal();
    } catch (err) {
      setIndexNote(err instanceof Error ? err.message : String(err));
    } finally {
      setIndexRunning(false);
    }
  };

  const copyText = async () => {
    if (!textBody) {
      return;
    }
    try {
      await navigator.clipboard.writeText(textBody);
      setTextTitle((prev) => (prev.endsWith(' · 已复制') ? prev : `${prev} · 已复制`));
    } catch {
      /* ignore */
    }
  };

  return (
    <>
      <div className="cursor-console-ops" ref={menuRef}>
        {isAgent && (
          <button
            type="button"
            className={`cursor-context-chip${contextPct !== null && contextPct >= 85 ? ' is-warn' : ''}`}
            onClick={openContext}
            title="查看上下文用量与压缩"
          >
            {contextPct !== null ? `${contextPct}%` : '上下文'}
          </button>
        )}

        <div className="cursor-quick-menu-wrap">
          <button
            type="button"
            className={`cursor-quick-trigger${menuOpen ? ' is-open' : ''}`}
            onClick={() => setMenuOpen((v) => !v)}
            aria-expanded={menuOpen}
            aria-haspopup="menu"
          >
            <span className="cursor-quick-trigger-icon" aria-hidden>
              ⚡
            </span>
            更多
          </button>
          {menuOpen && (
            <div className="cursor-quick-menu cursor-quick-menu--compact" role="menu">
              {sandbox && (
                <div className="cursor-quick-menu-sandbox">
                  <span className="cursor-quick-menu-sandbox-label">沙箱</span>
                  <span
                    className={`cursor-quick-menu-sandbox-badge${sandbox.enabled ? ' is-on' : sandbox.active ? ' is-pending' : ''}`}
                  >
                    {sandboxLabel(sandbox)}
                  </span>
                </div>
              )}
              <div className="cursor-quick-menu-list">
                <button type="button" onClick={() => void runMetaText('/help', '帮助')}>
                  帮助
                </button>
                <button type="button" onClick={() => void runMetaText('/sessions', '会话列表')}>
                  会话列表
                </button>
                <button type="button" onClick={openIndex}>
                  代码索引
                </button>
                {!isAgent && (
                  <button type="button" onClick={openContext}>
                    上下文用量
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    onOpenLog?.();
                    closeMenu();
                  }}
                >
                  执行日志
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {modal === 'context' && (
        <OpsModalShell title="上下文" onClose={closeModal}>
          {contextLoading && !contextUsage && <ProgressStrip label="加载上下文用量…" />}
          {compressing && <ProgressStrip label="正在压缩历史，可能需要数十秒…" />}
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
                <div className="cursor-context-bar cursor-context-bar--lg">
                  <div
                    className={`cursor-context-bar-fill${contextUsage.pct >= 85 ? ' is-warn' : ''}`}
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
              <p className="cursor-ops-meta-line">
                消息 {contextUsage.message_count} · 工具 {contextUsage.tool_count}
                {contextUsage.mcp_tool_count > 0 ? ` · MCP ${contextUsage.mcp_tool_count}` : ''}
              </p>
              {contextUsage.budget_note && (
                <p className="cursor-ops-meta-line muted">{contextUsage.budget_note}</p>
              )}
              {isAgent && contextUsage.has_session && (
                <div className="cursor-ops-modal-actions">
                  <button
                    type="button"
                    className="cursor-ops-primary-btn"
                    disabled={compressing || contextLoading}
                    onClick={() => void compressHistory()}
                  >
                    {compressing ? '压缩中…' : '压缩历史'}
                  </button>
                  <button
                    type="button"
                    className="cursor-ops-secondary-btn"
                    disabled={compressing || contextLoading}
                    onClick={() => void runMetaText('/context', '上下文文本')}
                  >
                    导出文本
                  </button>
                </div>
              )}
              {isAgent && !contextUsage.has_session && (
                <p className="cursor-ops-meta-line muted">当前会话尚无多轮历史可压缩</p>
              )}
            </>
          )}
          {contextNote && <pre className="cursor-ops-note">{contextNote}</pre>}
        </OpsModalShell>
      )}

      {modal === 'index' && (
        <OpsModalShell title="代码索引" onClose={closeModal}>
          {(indexLoading || indexRunning) && !indexStatus && (
            <ProgressStrip label={indexRunning ? '索引构建中，请稍候…' : '加载索引状态…'} />
          )}
          {indexRunning && indexStatus && <ProgressStrip label="索引构建中，请稍候…" />}
          {indexStatus && (
            <>
              <div className="cursor-ops-index-grid">
                <div className="cursor-ops-index-stat">
                  <span>状态</span>
                  <strong>{indexStatus.exists ? '已建立' : '未建立'}</strong>
                </div>
                <div className="cursor-ops-index-stat">
                  <span>Chunks</span>
                  <strong>{indexStatus.chunk_count.toLocaleString()}</strong>
                </div>
                <div className="cursor-ops-index-stat">
                  <span>向量维度</span>
                  <strong>{indexStatus.vector_dim || '—'}</strong>
                </div>
                <div className="cursor-ops-index-stat">
                  <span>Manifest</span>
                  <strong>{indexStatus.manifest_files} 文件</strong>
                </div>
              </div>
              <p className="cursor-ops-meta-line muted">
                最近索引 {indexStatus.last_indexed_at || '无'} · Watch{' '}
                {indexStatus.watch_enabled ? '开启' : '关闭'}
              </p>
              <p className="cursor-ops-embed-line">{indexStatus.embedding}</p>
              <div className="cursor-ops-modal-actions">
                {(['incremental', 'full', 'rebuild'] as const).map((action) => (
                  <button
                    key={action}
                    type="button"
                    className="cursor-ops-secondary-btn"
                    disabled={indexRunning || indexLoading}
                    onClick={() => void runIndexAction(action)}
                  >
                    {action === 'incremental' ? '增量' : action === 'full' ? '全量' : '重建'}
                  </button>
                ))}
              </div>
            </>
          )}
          {indexNote && <pre className="cursor-ops-note">{indexNote}</pre>}
        </OpsModalShell>
      )}

      {modal === 'text' && (
        <OpsModalShell title={textTitle} onClose={closeModal} wide>
          {textLoading && <ProgressStrip label="加载中…" />}
          {!textLoading && textBody && (
            <>
              <pre className="cursor-ops-text">{textBody}</pre>
              <div className="cursor-ops-modal-actions">
                <button type="button" className="cursor-ops-secondary-btn" onClick={() => void copyText()}>
                  复制
                </button>
              </div>
            </>
          )}
        </OpsModalShell>
      )}
    </>
  );
});

export default ConsoleOps;
