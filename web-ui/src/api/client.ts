const BASE = '/api';

const SSE_RECONNECT_MS = 1500;

/** EventSource 断线后自动重连（切标签/网络抖动时恢复订阅）。 */
function subscribeReconnectingSSE(
  url: string,
  onEvent: (data: Record<string, unknown>) => void,
): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  const connect = () => {
    if (closed) {
      return;
    }
    es = new EventSource(url);
    es.onmessage = (ev) => {
      try {
        onEvent(JSON.parse(ev.data) as Record<string, unknown>);
      } catch {
        /* ignore */
      }
    };
    es.onerror = () => {
      es?.close();
      es = null;
      if (!closed) {
        retryTimer = setTimeout(connect, SSE_RECONNECT_MS);
      }
    };
  };

  connect();
  return () => {
    closed = true;
    if (retryTimer !== null) {
      clearTimeout(retryTimer);
    }
    es?.close();
  };
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    let message = text || res.statusText;
    try {
      const parsed = JSON.parse(text) as { detail?: string };
      if (typeof parsed.detail === 'string' && parsed.detail) {
        message = parsed.detail;
      }
    } catch {
      // 非 JSON 响应，保留原始文本
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export interface Workspace {
  slug: string;
  path: string;
  session_count: number;
  plan_count?: number;
  updated_at: string | null;
}

export interface TreeNode {
  kind: 'agent' | 'subagent';
  thread_id: string;
  title: string;
  title_full?: string;
  updated_at?: string | null;
  /** 仅前端：新建会话尚未出现在 tree API 前保留侧栏占位 */
  _optimistic?: boolean;
  phase?: string;
  task_id?: string;
  status?: string;
  children: TreeNode[];
}

export interface Capabilities {
  builtin_tools: Array<{ name: string; description: string }>;
  mcp_tools: Array<{ name: string; description: string }>;
  mcp_summary: string;
  mcp_errors?: string[];
  mcp_loading?: boolean;
  mcp_servers: Array<{
    name: string;
    command: string;
    enabled: boolean;
    status?: 'ok' | 'error' | 'loading' | 'idle';
    error?: string | null;
    tool_count?: number;
  }>;
  skills: Array<{ name: string; description: string; scope: string; scope_label?: string; path: string; active?: boolean }>;
  rules: Array<{ id: string; description: string; scope: string; scope_label?: string; path: string; forced?: boolean; disabled?: boolean }>;
  commands: Array<{ name: string; description: string; requires_write: boolean }>;
  web_search_enabled: boolean;
  trace_mode: string;
  sandbox?: {
    active: boolean;
    enabled: boolean;
    backend: string | null;
    mode: string;
    network: string;
    cli_override?: boolean | null;
  };
  context_state?: {
    active_skills: string[];
    disabled_rules: string[];
    forced_rules: string[];
  };
}

export interface MemoryHit {
  memory_id: string;
  kind: string;
  content: string;
  score?: number;
  hit_count?: number;
  confidence?: number;
  source?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  last_hit_at?: string;
  content_hash?: string;
}

export interface MemoryStatus {
  enabled: boolean;
  user_id: string;
  workspace_key: string;
  workspace_slug: string;
  memory_root: string;
  counts: {
    active: number;
    pref: number;
    fact: number;
    proc: number;
  };
  settings?: {
    auto_recall_top_k: number;
    search_tool_top_k: number;
    auto_recall_min_score: number;
    memory_content_max_chars: number;
  };
}

export interface MemorySearchResult {
  enabled: boolean;
  mode: 'search' | 'browse' | 'disabled';
  query: string;
  top_k: number;
  min_score?: number;
  hits: MemoryHit[];
  filtered_below_min?: number;
  elapsed_ms?: number;
  count?: number;
  message?: string;
}

export interface ContextUsage {
  total: number;
  limit: number;
  ratio: number;
  pct: number;
  message_count: number;
  tool_count: number;
  mcp_tool_count: number;
  breakdown: Record<string, number>;
  budget_note: string;
  has_session: boolean;
}

export interface ContextSettingsSnapshot {
  compress_strategy: string;
  auto_compress_ratio: number;
  compress_during_react: boolean;
  incremental_tool_prune: boolean;
  keep_recent_tool_messages: number;
  dispatch_tool_chain_compress?: boolean;
  dispatch_keep_full_tool_messages?: number;
  tool_result_max_chars: number;
}

export interface ContextFixedSection {
  key: string;
  title: string;
  tokens: number;
  preview: string;
  truncated: boolean;
}

export interface ContextBreakdownSection {
  key: string;
  title: string;
  tokens: number;
  preview: string;
  truncated: boolean;
  messages?: ContextMessageInspect[];
  empty_hint?: string;
  from_disk?: boolean;
}

export interface ContextMessageInspect {
  index: number;
  role: string;
  kind: string;
  tokens: number;
  chars: number;
  preview: string;
  truncated: boolean;
  tool_name?: string;
}

export interface ContextDetail {
  usage: ContextUsage;
  settings: ContextSettingsSnapshot;
  compress_threshold: number;
  config_help: string;
  breakdown_sections: ContextBreakdownSection[];
  fixed_sections: ContextFixedSection[];
  stored_messages: ContextMessageInspect[];
  dispatch_messages: ContextMessageInspect[];
  dispatch_note: string;
}

export interface IndexProgress {
  running: boolean;
  action?: string | null;
  phase?: string | null;
  files_scanned?: number;
  files_skipped?: number;
  files_updated?: number;
  chunks_written?: number;
  files_total?: number | null;
  percent?: number | null;
  elapsed_sec?: number;
  error?: string | null;
  ok?: boolean;
  updated_at?: string;
}

export interface IndexStatus {
  exists: boolean;
  chunk_count: number;
  vector_dim: number;
  last_indexed_at: string | null;
  lance_path: string;
  manifest_files: number;
  sync_complete: boolean | null;
  watch_enabled: boolean;
  watch_with_agent: boolean;
  /** Web 进程内文件监听是否已启动 */
  watch_active?: boolean;
  /** 手动索引占用锁时 Watch 已暂停（非启动失败） */
  watch_paused?: boolean;
  embedding: string;
  max_files: number;
  /** 同步进度（后台 index 时轮询） */
  progress?: IndexProgress | null;
  indexing?: boolean;
}

export interface LlmModelOption {
  id: string;
  hint: string;
  rate: number | null;
  supports_thinking: boolean;
  current: boolean;
}

export interface LlmSettings {
  model: string;
  model_runtime_override: string | null;
  models: LlmModelOption[];
  models_source: string;
  provider_label: string;
  thinking: {
    supported: boolean;
    enabled: boolean;
    runtime_override: boolean | null;
  };
}

export interface FileChangeItem {
  path: string;
  has_snapshot: boolean;
  kind: 'modified' | 'created';
  edit_count: number;
}

export interface FileChangesSummary {
  session_id: string;
  paths: FileChangeItem[];
  total: number;
  can_undo: boolean;
}

export interface MessageItem {
  type: string;
  content: unknown;
  /** 后端提取的用户可见正文（thinking 块已剥离） */
  display_text?: string;
  /** 用户消息附带的图片（image_ref 预览） */
  images?: Array<{ media_type: string; url?: string; id?: string }>;
  /** 内部消息种类，如 think_nudge */
  kind?: string;
  name?: string;
  tool_calls?: unknown;
  raw?: Record<string, unknown>;
}

export interface SessionSummary {
  thread_id: string;
  title: string;
  updated_at: string | null;
  message_count: number;
  has_edits: boolean;
}

export interface SlashCatalogItem {
  name: string;
  description: string;
  category: string;
  badge: string;
  insert_text: string;
  origin?: string;
}

export type SSEHandler = (event: Record<string, unknown>) => void;

export async function consumeSSE(
  url: string,
  body: unknown,
  onEvent: SSEHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}${url}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(await res.text());
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) {
        continue;
      }
      try {
        const data = JSON.parse(line.slice(5).trim()) as Record<string, unknown>;
        onEvent(data);
      } catch {
        /* ignore */
      }
    }
  }
}

export const api = {
  workspaces: () => fetchJson<{ workspaces: Workspace[] }>('/workspaces'),
  registerWorkspace: (path: string) =>
    fetchJson<Workspace>('/workspaces/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }),
  pickWorkspaceDirectory: (initialPath = '') =>
    fetchJson<{ path: string | null; cancelled: boolean }>('/workspaces/pick-directory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ initial_path: initialPath }),
    }),
  dismissWorkspace: (slug: string) =>
    fetchJson<{ ok: boolean; slug: string; message: string }>(`/workspaces/${slug}/recent`, {
      method: 'DELETE',
    }),
  touchWorkspace: (slug: string) =>
    fetchJson<{ ok: boolean; slug: string }>(`/workspaces/${slug}/touch`, {
      method: 'POST',
    }),
  tree: (slug: string) =>
    fetchJson<{ agents: TreeNode[] }>(`/workspaces/${slug}/tree`),
  sessions: (slug: string) =>
    fetchJson<{ sessions: SessionSummary[] }>(`/workspaces/${slug}/sessions`),
  renameSession: (slug: string, threadId: string, title: string) =>
    fetchJson<{ ok: boolean; title: string; message: string }>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/title`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      },
    ),
  session: (slug: string, threadId: string) =>
    fetchJson<{ thread_id: string; meta: Record<string, unknown>; title: string; message_total: number }>(
      `/workspaces/${slug}/sessions/${threadId}`,
    ),
  capabilities: (slug: string, allowWrite = false) =>
    fetchJson<Capabilities>(`/workspaces/${slug}/capabilities?allow_write=${allowWrite}`),
  llmSettings: (slug: string) =>
    fetchJson<LlmSettings>(`/workspaces/${slug}/llm-settings`),
  setLlmSettings: (
    slug: string,
    body: {
      model?: string;
      thinking_enabled?: boolean;
      reset_model?: boolean;
      reset_thinking?: boolean;
    },
  ) =>
    fetchJson<LlmSettings>(`/workspaces/${slug}/llm-settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  fileChanges: (slug: string, threadId: string) =>
    fetchJson<FileChangesSummary>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/file-changes`,
    ),
  undoFiles: (slug: string, threadId: string, target = 'all') =>
    fetchJson<{
      ok: boolean;
      summary: { restored: number; deleted: number; skipped: number; failed: number };
      results: Array<{ rel_path: string; action: string; detail: string }>;
      changes: FileChangesSummary;
    }>(`/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/undo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target }),
    }),
  fileDiff: (slug: string, threadId: string, path: string) =>
    fetchJson<{ path: string; diff: string }>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/diff?path=${encodeURIComponent(path)}`,
    ),
  reviewSession: (slug: string, threadId: string, topic = '') =>
    fetchJson<{ ok: boolean; message?: string; review_path?: string; summary?: string }>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/review`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic }),
      },
    ),
  skillDetail: (slug: string, name: string) =>
    fetchJson<{
      name: string;
      description: string;
      scope: string;
      scope_label: string;
      path: string;
      body: string;
    }>(`/workspaces/${slug}/catalog/skill/${encodeURIComponent(name)}`),
  ruleDetail: (slug: string, ruleId: string) =>
    fetchJson<{
      id: string;
      description: string;
      scope: string;
      scope_label: string;
      path: string;
      body: string;
    }>(`/workspaces/${slug}/catalog/rule/${encodeURIComponent(ruleId)}`),
  setTraceMode: (slug: string, mode: string) =>
    fetchJson<{ mode: string }>(`/workspaces/${slug}/trace-mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    }),
  slashCatalog: (slug: string) =>
    fetchJson<{ items: SlashCatalogItem[] }>(`/workspaces/${slug}/slash-catalog`),
  metaCommand: (
    slug: string,
    command: string,
    allowWrite: boolean,
    threadId = '',
  ) =>
    fetchJson<{
      handled: boolean;
      registered: boolean;
      output: string;
      trace_mode: string;
      display_mode?: 'modal' | 'agent';
    }>(`/workspaces/${slug}/meta`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        command,
        allow_write: allowWrite,
        thread_id: threadId,
      }),
    }),
  createSession: (slug: string, kind: 'agent', goal = '') =>
    fetchJson<{ thread_id: string; kind: string }>(`/workspaces/${slug}/sessions/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, goal }),
    }),
  prewarmSession: (slug: string, threadId: string, allowWrite = true) =>
    fetchJson<{ ok: boolean; thread_id: string; duration_sec?: number }>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/warm?allow_write=${allowWrite ? '1' : '0'}`,
      { method: 'POST' },
    ),
  warmRecentSessions: (slug: string, allowWrite = true) =>
    fetchJson<{ ok: boolean; queued: number; limit: number; background?: boolean }>(
      `/workspaces/${slug}/sessions/warm-recent?allow_write=${allowWrite ? '1' : '0'}`,
      { method: 'POST' },
    ),
  deleteSession: (slug: string, threadId: string) =>
    fetchJson<{
      thread_id: string;
      kind: string;
      ok: boolean;
      removed_paths: string[];
      related_removed: string[];
      error?: string;
    }>(`/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}`, {
      method: 'DELETE',
    }),
  deleteSessions: (slug: string, threadIds: string[]) =>
    fetchJson<{
      ok: boolean;
      success_count: number;
      failure_count: number;
      results: Array<{
        thread_id: string;
        ok: boolean;
        removed_paths: string[];
        related_removed: string[];
        error?: string;
      }>;
    }>(`/workspaces/${slug}/sessions/batch-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_ids: threadIds }),
    }),
  messages: (slug: string, threadId: string, offset = 0, limit = 200, tail = true) =>
    fetchJson<{ messages: MessageItem[]; total: number }>(
      `/workspaces/${slug}/sessions/${threadId}/messages?offset=${offset}&limit=${limit}&tail=${tail ? '1' : '0'}`,
    ),
  sessionMeta: (slug: string, threadId: string) =>
    fetchJson<{
      thread_id: string;
      title: string;
      message_total: number;
      allow_write?: boolean;
      running?: boolean;
      lock: { owner: string; since: number } | null;
    }>(`/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}`),
  touchSession: (slug: string, threadId: string) =>
    fetchJson<{ thread_id: string; updated_at?: string | null }>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/touch`,
      { method: 'POST' },
    ),
  sessionTrace: (slug: string, threadId: string) =>
    fetchJson<{
      log_lines: string[];
      steps: Record<string, unknown>[];
      turns?: Record<string, unknown>[];
      live_ts?: string;
    }>(`/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/trace`),
  startAgentChat: (
    slug: string,
    threadId: string,
    message: string,
    allowWrite: boolean,
    imageFiles?: File[],
  ) => {
    const form = new FormData();
    form.append('message', message);
    form.append('allow_write', allowWrite ? 'true' : 'false');
    for (const file of imageFiles ?? []) {
      form.append('images', file);
    }
    return fetchJson<{
      ok: boolean;
      thread_id: string;
      images?: Array<{ id: string; media_type: string; url: string }>;
    }>(`/workspaces/${slug}/sessions/${threadId}/chat`, {
      method: 'POST',
      body: form,
    });
  },
  abortAgentChat: (slug: string, threadId: string) =>
    fetchJson<{ ok: boolean; message: string }>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/abort`,
      { method: 'POST' },
    ),
  setWebSearch: (
    slug: string,
    enabled: boolean,
    threadId = '',
    allowWrite = false,
  ) =>
    fetchJson<{ enabled: boolean; message: string }>(`/workspaces/${slug}/web-search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled, thread_id: threadId, allow_write: allowWrite }),
    }),
  setSandbox: (
    slug: string,
    enabled: boolean,
    threadId = '',
    allowWrite = false,
  ) =>
    fetchJson<{ sandbox: Capabilities['sandbox']; message: string }>(`/workspaces/${slug}/sandbox`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled, thread_id: threadId, allow_write: allowWrite }),
    }),
  setWriteMode: (slug: string, enabled: boolean, threadId = '') =>
    fetchJson<{ enabled: boolean; message: string }>(`/workspaces/${slug}/write-mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled, thread_id: threadId }),
    }),
  contextUsage: (slug: string, allowWrite = false, threadId = '') =>
    fetchJson<ContextUsage>(
      `/workspaces/${slug}/context?allow_write=${allowWrite}&thread_id=${encodeURIComponent(threadId)}`,
    ),
  contextDetail: (
    slug: string,
    allowWrite = false,
    threadId = '',
    maxPreviewChars = 6000,
  ) =>
    fetchJson<ContextDetail>(
      `/workspaces/${slug}/context/detail?allow_write=${allowWrite}&thread_id=${encodeURIComponent(threadId)}&max_preview_chars=${maxPreviewChars}`,
    ),
  compressContext: (slug: string, threadId: string, allowWrite = false) =>
    fetchJson<{ ok: boolean; compressed: boolean; message: string; archive_path?: string }>(
      `/workspaces/${slug}/compress`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, allow_write: allowWrite }),
      },
    ),
  indexStatus: (slug: string) => fetchJson<IndexStatus>(`/workspaces/${slug}/index-status`),
  runIndex: (slug: string, action: 'full' | 'incremental' | 'rebuild' | 'dry-run' | 'status') =>
    fetchJson<{
      ok: boolean;
      started?: boolean;
      exit_code: number | null;
      action: string;
      log_path: string | null;
    }>(`/workspaces/${slug}/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    }),
  toggleSkill: (slug: string, name: string, active: boolean) =>
    fetchJson<{ ok: boolean; name: string; active: boolean; active_skills: string[]; message: string }>(
      `/workspaces/${slug}/catalog/skill/${encodeURIComponent(name)}/toggle`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active }),
      },
    ),
  toggleRule: (slug: string, ruleId: string, enabled: boolean) =>
    fetchJson<{ ok: boolean; id: string; forced: boolean; disabled: boolean; message: string }>(
      `/workspaces/${slug}/catalog/rule/${encodeURIComponent(ruleId)}/toggle`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      },
    ),
  deleteEmptySessions: (slug: string) =>
    fetchJson<{ ok: boolean; deleted: number; failed?: number; message: string }>(
      `/workspaces/${slug}/sessions/delete-empty`,
      { method: 'POST' },
    ),
  subscribeSessionEvents: (
    slug: string,
    threadId: string,
    onEvent: (data: Record<string, unknown>) => void,
  ): (() => void) =>
    subscribeReconnectingSSE(
      `${BASE}/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/events`,
      onEvent,
    ),
  codeSearch: (
    slug: string,
    query: string,
    opts: { mode?: 'parallel' | 'semantic'; top_k?: number; path_prefix?: string } = {},
  ) =>
    fetchJson<{ query: string; mode: string; top_k: number; text: string; lines: string[]; count: number }>(
      `/workspaces/${slug}/code-search`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          mode: opts.mode || 'parallel',
          top_k: opts.top_k ?? 8,
          path_prefix: opts.path_prefix || '.',
        }),
      },
    ),
  memoryStatus: (slug: string) =>
    fetchJson<MemoryStatus>(`/workspaces/${slug}/memory-status`),
  memorySearch: (
    slug: string,
    query: string,
    opts: {
      top_k?: number;
      min_score?: number;
      kind?: string;
    } = {},
  ) =>
    fetchJson<MemorySearchResult>(`/workspaces/${slug}/memory-search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: opts.top_k ?? 20,
        min_score: opts.min_score ?? 0,
        kind: opts.kind ?? null,
      }),
    }),
  executionLog: (slug: string, limit = 30) =>
    fetchJson<{ path: string; lines: string[]; count: number }>(
      `/workspaces/${slug}/execution-log?limit=${limit}`,
    ),
  purgeExecutionLog: (slug: string) =>
    fetchJson<{ ok: boolean; message: string }>(`/workspaces/${slug}/execution-log/purge`, {
      method: 'POST',
    }),
};
