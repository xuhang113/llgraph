const BASE = '/api';

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
  plan_count: number;
  updated_at: string | null;
}

export interface TreeNode {
  kind: 'agent' | 'plan' | 'worker';
  thread_id: string;
  title: string;
  updated_at?: string | null;
  phase?: string;
  plan_id?: string;
  task_id?: string;
  status?: string;
  children: TreeNode[];
}

export interface Capabilities {
  builtin_tools: Array<{ name: string; description: string }>;
  mcp_tools: Array<{ name: string; description: string }>;
  mcp_summary: string;
  mcp_servers: Array<{ name: string; command: string; enabled: boolean }>;
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
  embedding: string;
  max_files: number;
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

export interface PlanFileChangesGroup extends FileChangesSummary {
  task_id: string;
  title: string;
  thread_id: string;
}

export interface PlanFileChangesSummary {
  plan_thread_id: string;
  groups: PlanFileChangesGroup[];
  total: number;
  can_undo: boolean;
}

export interface MessageItem {
  type: string;
  content: unknown;
  /** 后端提取的用户可见正文（thinking 块已剥离） */
  display_text?: string;
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

export interface PlanSummary {
  thread_id: string;
  plan_id: string;
  title: string;
  phase: string;
  tasks_done: number;
  tasks_total: number;
  updated_at: string | null;
}

export interface SlashCatalogItem {
  name: string;
  description: string;
  category: string;
  badge: string;
  insert_text: string;
  origin?: string;
}

export interface WorkflowNode {
  id: string;
  status: string;
}

export interface WorkflowTask {
  id: string;
  title: string;
  status: string;
  worker_node_status?: string;
  depends_on?: string[];
}

export interface WorkerDetail {
  thread_id: string;
  task_id: string;
  worker_thread_id: string;
  task: Record<string, unknown>;
  result: Record<string, unknown> | null;
  messages: MessageItem[];
  message_total: number;
  edits: Array<Record<string, unknown>>;
}

export interface PlanDetail {
  thread_id: string;
  plan_id: string;
  title: string;
  goal: string;
  phase: string;
  final_report: string | null;
  error: string | null;
  tasks: Array<Record<string, unknown>>;
  workflow_snapshot: {
    nodes?: WorkflowNode[];
    tasks?: WorkflowTask[];
    synthesize_depends_on?: string[];
    graph_definition?: Record<string, unknown>;
  };
  updated_at?: string | null;
  job?: { running?: boolean; error?: string | null };
  plan_state?: {
    user_messages?: string[];
    revision_note?: string | null;
  };
}

export interface SurveySpec {
  title: string;
  questions: Array<{
    id: string;
    prompt: string;
    options: string[];
    default_index: number;
    default_indices?: number[];
    allow_free_text?: boolean;
    step_label?: string;
    option_hints?: string[];
    multi_select?: boolean;
  }>;
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
    fetchJson<{ agents: TreeNode[]; plans: TreeNode[] }>(`/workspaces/${slug}/tree`),
  sessions: (slug: string) =>
    fetchJson<{ sessions: SessionSummary[] }>(`/workspaces/${slug}/sessions`),
  plans: (slug: string) =>
    fetchJson<{ plans: PlanSummary[] }>(`/workspaces/${slug}/plans`),
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
  planFileChanges: (slug: string, planThreadId: string) =>
    fetchJson<PlanFileChangesSummary>(
      `/workspaces/${slug}/plans/${encodeURIComponent(planThreadId)}/file-changes`,
    ),
  planUndo: (slug: string, planThreadId: string, target = 'all', taskId?: string) =>
    fetchJson<{
      ok: boolean;
      summary: { restored: number; deleted: number; skipped: number; failed: number };
      plan_changes: PlanFileChangesSummary;
    }>(`/workspaces/${slug}/plans/${encodeURIComponent(planThreadId)}/undo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target, task_id: taskId || null }),
    }),
  planReview: (slug: string, planThreadId: string, topic = '') =>
    fetchJson<{
      ok: boolean;
      message?: string;
      reviews?: Array<{ task_id?: string; review_path?: string; summary?: string }>;
    }>(`/workspaces/${slug}/plans/${encodeURIComponent(planThreadId)}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic }),
    }),
  formatSurveyAnswers: (slug: string, answers: Record<string, string>, allowWrite: boolean) =>
    fetchJson<{ message: string }>(`/workspaces/${slug}/survey/format`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers, allow_write: allowWrite }),
    }),
  resolveSurvey: (slug: string, text: string) =>
    fetchJson<{ survey: SurveySpec | null }>(`/workspaces/${slug}/survey/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }),
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
    }>(`/workspaces/${slug}/meta`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        command,
        allow_write: allowWrite,
        thread_id: threadId,
      }),
    }),
  createSession: (slug: string, kind: 'agent' | 'plan', goal = '') =>
    fetchJson<{ thread_id: string; kind: string }>(`/workspaces/${slug}/sessions/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, goal }),
    }),
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
  messages: (slug: string, threadId: string, offset = 0, limit = 200) =>
    fetchJson<{ messages: MessageItem[]; total: number }>(
      `/workspaces/${slug}/sessions/${threadId}/messages?offset=${offset}&limit=${limit}`,
    ),
  lastTrace: (slug: string, threadId: string) =>
    fetchJson<{ log_lines: string[]; steps: Record<string, unknown>[] }>(
      `/workspaces/${slug}/sessions/${encodeURIComponent(threadId)}/last-trace`,
    ),
  plan: (slug: string, threadId: string) =>
    fetchJson<PlanDetail>(`/workspaces/${slug}/plans/${threadId}`),
  worker: (slug: string, threadId: string, taskId: string) =>
    fetchJson<WorkerDetail>(`/workspaces/${slug}/plans/${threadId}/tasks/${taskId}`),
  agentChat: (slug: string, threadId: string, message: string, allowWrite: boolean, onEvent: SSEHandler, signal?: AbortSignal) =>
    consumeSSE(
      `/workspaces/${slug}/sessions/${threadId}/chat`,
      { message, allow_write: allowWrite },
      onEvent,
      signal,
    ),
  planStart: (slug: string, threadId: string, message: string, allowWrite: boolean, onEvent: SSEHandler, signal?: AbortSignal) =>
    consumeSSE(
      `/workspaces/${slug}/plans/${threadId}/start`,
      { message, allow_write: allowWrite },
      onEvent,
      signal,
    ),
  planConfirm: (slug: string, threadId: string, body: Record<string, unknown>, onEvent: SSEHandler, signal?: AbortSignal) =>
    consumeSSE(`/workspaces/${slug}/plans/${threadId}/confirm`, body, onEvent, signal),
  planContinue: (slug: string, threadId: string, allowWrite: boolean, onEvent: SSEHandler, signal?: AbortSignal) =>
    consumeSSE(
      `/workspaces/${slug}/plans/${threadId}/continue`,
      { message: '', allow_write: allowWrite },
      onEvent,
      signal,
    ),
  planDiscuss: (slug: string, threadId: string, message: string, onEvent: SSEHandler, signal?: AbortSignal) =>
    consumeSSE(
      `/workspaces/${slug}/plans/${threadId}/discuss`,
      { message, allow_write: false },
      onEvent,
      signal,
    ),
  planTaskRunnable: (slug: string, threadId: string, taskId: string) =>
    fetchJson<{ ok: boolean; message: string; missing_deps: string[] }>(
      `/workspaces/${slug}/plans/${threadId}/tasks/${taskId}/runnable`,
    ),
  planRunTask: (
    slug: string,
    threadId: string,
    taskId: string,
    allowWrite: boolean,
    onEvent: SSEHandler,
    signal?: AbortSignal,
  ) =>
    consumeSSE(
      `/workspaces/${slug}/plans/${threadId}/tasks/${taskId}/run`,
      { message: '', allow_write: allowWrite },
      onEvent,
      signal,
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
  contextUsage: (slug: string, allowWrite = false, threadId = '') =>
    fetchJson<ContextUsage>(
      `/workspaces/${slug}/context?allow_write=${allowWrite}&thread_id=${encodeURIComponent(threadId)}`,
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
    fetchJson<{ ok: boolean; exit_code: number; action: string; log_path: string | null }>(
      `/workspaces/${slug}/index`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      },
    ),
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
  subscribePlanEvents: (
    slug: string,
    threadId: string,
    onEvent: (data: Record<string, unknown>) => void,
  ): (() => void) => {
    const es = new EventSource(
      `${BASE}/workspaces/${slug}/plans/${encodeURIComponent(threadId)}/events`,
    );
    es.onmessage = (ev) => {
      try {
        onEvent(JSON.parse(ev.data) as Record<string, unknown>);
      } catch {
        /* ignore */
      }
    };
    return () => es.close();
  },
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
          top_k: opts.top_k ?? 15,
          path_prefix: opts.path_prefix || '.',
        }),
      },
    ),
  executionLog: (slug: string, limit = 30) =>
    fetchJson<{ path: string; lines: string[]; count: number }>(
      `/workspaces/${slug}/execution-log?limit=${limit}`,
    ),
  purgeExecutionLog: (slug: string) =>
    fetchJson<{ ok: boolean; message: string }>(`/workspaces/${slug}/execution-log/purge`, {
      method: 'POST',
    }),
};
