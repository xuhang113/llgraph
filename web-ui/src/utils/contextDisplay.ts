export const CONTEXT_BREAKDOWN_LABELS: Record<string, string> = {
  system_prompt: 'System prompt',
  tool_definitions: 'Tool definitions',
  rules: 'Rules',
  skills: 'Skills',
  mcp: 'MCP',
  markdowns_index: 'Markdowns index',
  summarized_conversation: 'Summarized conversation',
  conversation: 'Conversation',
};

/** 与后端 _BUILTIN_META_COMMAND_NAMES 对齐（首 token，不含 /） */
const BUILTIN_META_TOKENS = new Set([
  'index',
  'survey',
  'paste',
  'p',
  'watch',
  'write',
  'web',
  'model',
  'config',
  'session',
  'sessionid',
  'session-id',
  'sessions',
  'plan',
  'help',
  'h',
  'compress',
  'context',
  'tools',
  'review',
  'commands',
  'trace',
  'log',
  'rule',
  'skill',
  'changes',
  'undo',
  'diff',
  'agent',
]);

const BUILTIN_META_TITLES: Record<string, string> = {
  help: '帮助',
  h: '帮助',
  trace: 'Trace',
  tools: '工具列表',
  commands: '自定义命令',
  log: '日志',
  rule: 'Rules',
  skill: 'Skills',
  sessions: '会话列表',
  session: '会话管理',
  sessionid: '会话 ID',
  'session-id': '会话 ID',
  context: '上下文文本',
  compress: '压缩历史',
  config: '配置路径',
  model: '模型',
  web: '联网搜索',
  write: '写权限',
  plan: 'Plan',
  survey: '梳理向导',
  paste: '多行粘贴',
  p: '多行粘贴',
  watch: '索引 Watch',
  review: '代码评审',
  index: '代码索引',
  changes: '文件变更',
  undo: '撤销',
  diff: 'Diff',
  agent: 'Agent 模式',
};

export type SlashMetaModalRoute =
  | { kind: 'context' }
  | { kind: 'index' }
  | { kind: 'meta'; command: string; title: string };

export function formatContextTokens(n: number): string {
  if (n >= 10_000) {
    return `~${(n / 1000).toFixed(1)}K`;
  }
  if (n >= 1000) {
    return `~${(n / 1000).toFixed(1)}K`;
  }
  return `~${n}`;
}

/** 解析斜杠命令首 token（不含 /，小写）。 */
export function parseSlashMetaToken(text: string): string | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('/')) {
    return null;
  }
  const head = trimmed.split(/\s+/)[0] ?? '';
  const token = head.slice(1).toLowerCase();
  return token || null;
}

export function isBuiltinMetaCommand(text: string): boolean {
  const token = parseSlashMetaToken(text);
  return token !== null && BUILTIN_META_TOKENS.has(token);
}

export function metaCommandModalTitle(text: string): string {
  const token = parseSlashMetaToken(text);
  if (!token) {
    return '命令输出';
  }
  return BUILTIN_META_TITLES[token] ?? `/${token}`;
}

/**
 * Web UI 斜杠元命令是否应走弹窗（不写入对话区）。
 * 自定义 .llgraph/commands（prompt 类）返回 null，由 meta API display_mode 决定。
 */
export function resolveSlashMetaModalRoute(text: string): SlashMetaModalRoute | null {
  const raw = text.trim();
  if (!raw.startsWith('/')) {
    return null;
  }
  const lower = raw.toLowerCase();
  const token = parseSlashMetaToken(raw);

  if (token === 'compress') {
    return { kind: 'context' };
  }

  if (token === 'index') {
    const sub = lower.replace(/^\/index\s*/, '').trim();
    if (!sub || sub === 'status' || sub === 'help' || sub === '?') {
      return { kind: 'index' };
    }
    return { kind: 'meta', command: raw, title: '代码索引' };
  }

  if (token && BUILTIN_META_TOKENS.has(token)) {
    return {
      kind: 'meta',
      command: raw,
      title: metaCommandModalTitle(raw),
    };
  }

  return null;
}

/** @deprecated 使用 resolveSlashMetaModalRoute */
export function matchInformationalMetaCommand(
  text: string,
): { kind: 'context' } | { kind: 'index' } | { kind: 'meta'; command: string; title: string } | null {
  const route = resolveSlashMetaModalRoute(text);
  if (!route) {
    return null;
  }
  if (route.kind === 'context') {
    return { kind: 'context' };
  }
  if (route.kind === 'index') {
    return { kind: 'index' };
  }
  return route;
}
