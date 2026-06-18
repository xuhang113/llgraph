import { useEffect, useRef, useState } from 'react';
import type { TreeNode, Workspace } from '../../api/client';

interface Props {
  slug: string;
  workspaces: Workspace[];
  agents: TreeNode[];
  plans: TreeNode[];
  selectedId: string | null;
  catalogOpen: 'skills' | 'rules' | 'tools' | null;
  multiSelectMode: boolean;
  selectedSessionIds: ReadonlySet<string>;
  busy?: boolean;
  onSlugChange: (slug: string) => void;
  onOpenWorkspace: () => Promise<void>;
  onDismissWorkspace: (slug: string) => void;
  onSelect: (node: TreeNode) => void;
  onNewAgent: () => void;
  onNewPlan: () => void;
  onDelete: (node: TreeNode) => void;
  onRename: (node: TreeNode, title: string) => Promise<void>;
  onCatalogOpen: (kind: 'skills' | 'rules' | 'tools') => void;
  onCodeSearch?: () => void;
  onEnterMultiSelect: () => void;
  onExitMultiSelect: () => void;
  onToggleSessionSelect: (threadId: string) => void;
  onSelectAllSessions: (nodes: TreeNode[]) => void;
  onBatchDelete: () => void;
  onDeleteEmpty?: () => void;
}

type SessionGroupKind = 'agent' | 'plan';

interface ContextMenuState {
  x: number;
  y: number;
  group: SessionGroupKind;
}

function displayLabel(node: TreeNode): string {
  if (node.title && node.title !== node.thread_id) {
    return node.title;
  }
  return node.kind === 'agent' || node.kind === 'plan' ? '未命名会话' : node.thread_id;
}

function SessionRow({
  node,
  selectedId,
  multiSelectMode,
  selectedSessionIds,
  onSelect,
  onDelete,
  onRename,
  onToggleSessionSelect,
  depth = 0,
}: {
  node: TreeNode;
  selectedId: string | null;
  multiSelectMode: boolean;
  selectedSessionIds: ReadonlySet<string>;
  onSelect: (n: TreeNode) => void;
  onDelete: (n: TreeNode) => void;
  onRename: (node: TreeNode, title: string) => Promise<void>;
  onToggleSessionSelect: (threadId: string) => void;
  depth?: number;
}) {
  const label = displayLabel(node);
  const selected =
    selectedId === node.thread_id ||
    (node.kind === 'plan' &&
      !!selectedId &&
      selectedId.startsWith(`${node.thread_id}:worker:`));
  const checked = selectedSessionIds.has(node.thread_id);
  const deletable = depth === 0 && (node.kind === 'agent' || node.kind === 'plan');
  const renamable = deletable && !multiSelectMode;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(label);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editing) {
      setDraft(label);
    }
  }, [label, editing]);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const startEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!renamable || saving) {
      return;
    }
    setDraft(label);
    setEditing(true);
  };

  const cancelEdit = () => {
    setDraft(label);
    setEditing(false);
  };

  const commitEdit = async () => {
    const next = draft.trim();
    if (!next || next === label) {
      cancelEdit();
      return;
    }
    setSaving(true);
    try {
      await onRename(node, next);
      setEditing(false);
    } catch {
      inputRef.current?.focus();
    } finally {
      setSaving(false);
    }
  };

  const handleRowClick = () => {
    if (multiSelectMode && deletable) {
      onToggleSessionSelect(node.thread_id);
      return;
    }
    onSelect(node);
  };

  return (
    <>
      <div
        className={`cursor-session-row-wrap ${selected ? 'is-active' : ''}${
          multiSelectMode && checked ? ' is-checked' : ''
        }`}
      >
        {editing ? (
          <div className="cursor-session-rename" style={{ paddingLeft: `${12 + depth * 12}px` }}>
            <input
              ref={inputRef}
              className="cursor-session-rename-input"
              value={draft}
              disabled={saving}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                e.stopPropagation();
                if (e.key === 'Enter') {
                  e.preventDefault();
                  void commitEdit();
                } else if (e.key === 'Escape') {
                  e.preventDefault();
                  cancelEdit();
                }
              }}
              onBlur={() => {
                void commitEdit();
              }}
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        ) : (
          <button
            type="button"
            className={`cursor-session-row ${selected ? 'is-active' : ''}`}
            style={{ paddingLeft: `${12 + depth * 12}px` }}
            onClick={handleRowClick}
            onDoubleClick={renamable ? startEdit : undefined}
            title={renamable ? `${node.thread_id}（双击重命名）` : node.thread_id}
          >
            {multiSelectMode && deletable && (
              <input
                type="checkbox"
                className="cursor-session-check"
                checked={checked}
                readOnly
                tabIndex={-1}
                aria-label={`选择 ${label}`}
              />
            )}
            <span className="cursor-session-label">{label}</span>
          </button>
        )}
        {renamable && !editing && (
          <button
            type="button"
            className="cursor-session-rename-btn"
            title="重命名"
            aria-label="重命名"
            onClick={startEdit}
          >
            ✎
          </button>
        )}
        {deletable && !editing && !multiSelectMode && (
          <button
            type="button"
            className="cursor-session-delete"
            title="删除会话"
            aria-label="删除会话"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(node);
            }}
          >
            ×
          </button>
        )}
      </div>
      {node.kind !== 'plan' &&
        node.children?.map((child) => (
          <SessionRow
            key={child.thread_id}
            node={child}
            selectedId={selectedId}
            multiSelectMode={multiSelectMode}
            selectedSessionIds={selectedSessionIds}
            onSelect={onSelect}
            onDelete={onDelete}
            onRename={onRename}
            onToggleSessionSelect={onToggleSessionSelect}
            depth={depth + 1}
          />
        ))}
    </>
  );
}

function SessionGroup({
  groupKind,
  title,
  nodes,
  selectedId,
  multiSelectMode,
  selectedSessionIds,
  onSelect,
  onDelete,
  onRename,
  onToggleSessionSelect,
  onSelectAllSessions,
  onGroupContextMenu,
}: {
  groupKind: SessionGroupKind;
  title: string;
  nodes: TreeNode[];
  selectedId: string | null;
  multiSelectMode: boolean;
  selectedSessionIds: ReadonlySet<string>;
  onSelect: (node: TreeNode) => void;
  onDelete: (node: TreeNode) => void;
  onRename: (node: TreeNode, title: string) => Promise<void>;
  onToggleSessionSelect: (threadId: string) => void;
  onSelectAllSessions: (nodes: TreeNode[]) => void;
  onGroupContextMenu: (e: React.MouseEvent, group: SessionGroupKind) => void;
}) {
  const allSelected =
    nodes.length > 0 && nodes.every((n) => selectedSessionIds.has(n.thread_id));
  const selectedInGroup = nodes.filter((n) => selectedSessionIds.has(n.thread_id)).length;

  return (
    <details className="cursor-session-group" open>
      <summary
        className="cursor-group-summary"
        onContextMenu={(e) => onGroupContextMenu(e, groupKind)}
        title="右键：多选删除"
      >
        <span className="cursor-group-summary-main">
          {title}
          <span className="cursor-group-count">{nodes.length}</span>
          {multiSelectMode && selectedInGroup > 0 && (
            <span className="cursor-group-selected-count">已选 {selectedInGroup}</span>
          )}
        </span>
        {multiSelectMode && nodes.length > 0 && (
          <button
            type="button"
            className="cursor-group-select-all"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onSelectAllSessions(nodes);
            }}
          >
            {allSelected ? '取消全选' : '全选'}
          </button>
        )}
      </summary>
      {nodes.length === 0 ? (
        <div className="cursor-session-empty">暂无 {title} 会话</div>
      ) : (
        nodes.map((n) => (
          <SessionRow
            key={n.thread_id}
            node={n}
            selectedId={selectedId}
            multiSelectMode={multiSelectMode}
            selectedSessionIds={selectedSessionIds}
            onSelect={onSelect}
            onDelete={onDelete}
            onRename={onRename}
            onToggleSessionSelect={onToggleSessionSelect}
          />
        ))
      )}
    </details>
  );
}

export default function CursorSidebar({
  slug,
  workspaces,
  agents,
  plans,
  selectedId,
  catalogOpen,
  multiSelectMode,
  selectedSessionIds,
  busy = false,
  onSlugChange,
  onOpenWorkspace,
  onDismissWorkspace,
  onSelect,
  onNewAgent,
  onNewPlan,
  onDelete,
  onRename,
  onCatalogOpen,
  onCodeSearch,
  onEnterMultiSelect,
  onExitMultiSelect,
  onToggleSessionSelect,
  onSelectAllSessions,
  onBatchDelete,
  onDeleteEmpty,
}: Props) {
  const current = workspaces.find((w) => w.slug === slug);
  const selectedCount = selectedSessionIds.size;
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [wsPickerOpen, setWsPickerOpen] = useState(false);

  const currentLabel = current
    ? current.path.split('/').filter(Boolean).pop() || current.slug
    : '选择工作区';

  useEffect(() => {
    if (!contextMenu) {
      return;
    }
    const close = () => setContextMenu(null);
    window.addEventListener('click', close);
    window.addEventListener('scroll', close, true);
    return () => {
      window.removeEventListener('click', close);
      window.removeEventListener('scroll', close, true);
    };
  }, [contextMenu]);

  const handleGroupContextMenu = (e: React.MouseEvent, group: SessionGroupKind) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({ x: e.clientX, y: e.clientY, group });
  };

  return (
    <aside className="cursor-sidebar">
      <div className="cursor-sidebar-top">
        <span className="cursor-brand">llgraph</span>
        <div className="cursor-new-actions">
          <button type="button" className="cursor-btn-primary" onClick={onNewAgent} disabled={busy}>
            New Agent
          </button>
          <button type="button" className="cursor-btn-ghost" onClick={onNewPlan} disabled={busy}>
            New Plan
          </button>
        </div>
      </div>

      <nav className="cursor-catalog-nav">
        <button
          type="button"
          className={`cursor-catalog-nav-btn${catalogOpen === 'skills' ? ' is-active' : ''}`}
          onClick={() => onCatalogOpen('skills')}
          disabled={!slug}
        >
          Skills
        </button>
        <button
          type="button"
          className={`cursor-catalog-nav-btn${catalogOpen === 'rules' ? ' is-active' : ''}`}
          onClick={() => onCatalogOpen('rules')}
          disabled={!slug}
        >
          Rules
        </button>
        <button
          type="button"
          className={`cursor-catalog-nav-btn${catalogOpen === 'tools' ? ' is-active' : ''}`}
          onClick={() => onCatalogOpen('tools')}
          disabled={!slug}
        >
          工具
        </button>
        <button
          type="button"
          className="cursor-catalog-nav-btn"
          onClick={() => onCodeSearch?.()}
          disabled={!slug}
        >
          搜代码
        </button>
      </nav>

      <div className="cursor-ws-section">
        <button
          type="button"
          className="cursor-ws-current"
          onClick={() => setWsPickerOpen((v) => !v)}
          title={current?.path || '切换工作区'}
        >
          <span className="cursor-ws-current-main">
            <span className="cursor-ws-current-name">{currentLabel}</span>
            <span className="cursor-ws-current-chevron">{wsPickerOpen ? '▾' : '▸'}</span>
          </span>
          {current?.path && (
            <span className="cursor-ws-current-path">{current.path}</span>
          )}
          {!current && (
            <span className="cursor-ws-current-path">点击展开选择工作空间</span>
          )}
        </button>

        {wsPickerOpen && (
          <div className="cursor-ws-picker-panel">
            <div className="cursor-ws-picker-label">最近工作区</div>
            <div className="cursor-ws-list">
              {workspaces.length === 0 ? (
                <div className="cursor-session-empty">暂无最近记录</div>
              ) : (
                workspaces.map((w) => {
                  const label = w.path.split('/').filter(Boolean).pop() || w.slug;
                  return (
                    <div
                      key={w.slug}
                      className={`cursor-ws-item-wrap${w.slug === slug ? ' is-active' : ''}`}
                    >
                      <button
                        type="button"
                        className={`cursor-ws-item${w.slug === slug ? ' is-active' : ''}`}
                        onClick={() => {
                          onSlugChange(w.slug);
                          setWsPickerOpen(false);
                        }}
                        title={w.path}
                      >
                        <span className="cursor-ws-item-slug">{label}</span>
                        <span className="cursor-ws-item-meta">
                          {w.session_count} agent · {w.plan_count} plan
                        </span>
                        {w.path && <span className="cursor-ws-item-path">{w.path}</span>}
                      </button>
                      <button
                        type="button"
                        className="cursor-ws-item-remove"
                        title="从最近列表移除（不删除会话）"
                        aria-label="从最近列表移除"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDismissWorkspace(w.slug);
                        }}
                      >
                        ×
                      </button>
                    </div>
                  );
                })
              )}
            </div>
            <div className="cursor-ws-picker-footer">
              <button
                type="button"
                className="cursor-ws-open-new"
                onClick={() => {
                  void onOpenWorkspace().then(() => setWsPickerOpen(false));
                }}
              >
                <span className="cursor-ws-open-new-icon" aria-hidden>
                  +
                </span>
                打开新工作空间
              </button>
            </div>
          </div>
        )}
      </div>

      {slug && (
        <div className="cursor-session-list">
          {multiSelectMode && (
            <div className="cursor-session-bulk-bar">
              <span className="cursor-session-bulk-count">已选 {selectedCount}</span>
              <button
                type="button"
                className="cursor-session-bulk-delete"
                disabled={busy || selectedCount === 0}
                onClick={onBatchDelete}
              >
                删除
              </button>
              <button
                type="button"
                className="cursor-session-bulk-cancel"
                disabled={busy}
                onClick={onExitMultiSelect}
              >
                取消
              </button>
            </div>
          )}
          <SessionGroup
            groupKind="agent"
            title="Agent"
            nodes={agents}
            selectedId={selectedId}
            multiSelectMode={multiSelectMode}
            selectedSessionIds={selectedSessionIds}
            onSelect={onSelect}
            onDelete={onDelete}
            onRename={onRename}
            onToggleSessionSelect={onToggleSessionSelect}
            onSelectAllSessions={onSelectAllSessions}
            onGroupContextMenu={handleGroupContextMenu}
          />
          <SessionGroup
            groupKind="plan"
            title="Plan"
            nodes={plans}
            selectedId={selectedId}
            multiSelectMode={multiSelectMode}
            selectedSessionIds={selectedSessionIds}
            onSelect={onSelect}
            onDelete={onDelete}
            onRename={onRename}
            onToggleSessionSelect={onToggleSessionSelect}
            onSelectAllSessions={onSelectAllSessions}
            onGroupContextMenu={handleGroupContextMenu}
          />
        </div>
      )}

      {contextMenu && (
        <div
          className="cursor-session-context-menu"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {multiSelectMode ? (
            <button
              type="button"
              className="cursor-session-context-item"
              onClick={() => {
                setContextMenu(null);
                onExitMultiSelect();
              }}
            >
              取消多选
            </button>
          ) : (
            <>
              <button
                type="button"
                className="cursor-session-context-item"
                disabled={busy || (agents.length === 0 && plans.length === 0)}
                onClick={() => {
                  setContextMenu(null);
                  onEnterMultiSelect();
                }}
              >
                多选删除
              </button>
              {onDeleteEmpty && (
                <button
                  type="button"
                  className="cursor-session-context-item"
                  disabled={busy}
                  onClick={() => {
                    setContextMenu(null);
                    void onDeleteEmpty();
                  }}
                >
                  删除空壳会话
                </button>
              )}
            </>
          )}
        </div>
      )}
    </aside>
  );
}
