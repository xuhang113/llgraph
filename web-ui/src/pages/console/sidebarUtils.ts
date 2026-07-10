import type { Dispatch, SetStateAction } from 'react';
import type { TreeNode } from '../../api/client';

export type SessionDateBucket = 'today' | 'yesterday' | 'previous7days' | 'older';

export const SESSION_DATE_BUCKET_LABELS: Record<SessionDateBucket, string> = {
  today: '今天',
  yesterday: '昨天',
  previous7days: '近 7 天',
  older: '更早',
};

const SESSION_DATE_BUCKET_ORDER: SessionDateBucket[] = [
  'today',
  'yesterday',
  'previous7days',
  'older',
];

function startOfLocalDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

/** 按 updated_at 归入 Cursor 式日期桶（本地时区）。 */
export function sessionDateBucket(updatedAt?: string | null): SessionDateBucket {
  if (!updatedAt) {
    return 'older';
  }
  const parsed = new Date(updatedAt);
  if (Number.isNaN(parsed.getTime())) {
    return 'older';
  }
  const now = new Date();
  const todayStart = startOfLocalDay(now);
  const yesterdayStart = new Date(todayStart);
  yesterdayStart.setDate(yesterdayStart.getDate() - 1);
  const weekStart = new Date(todayStart);
  weekStart.setDate(weekStart.getDate() - 7);

  if (parsed >= todayStart) {
    return 'today';
  }
  if (parsed >= yesterdayStart) {
    return 'yesterday';
  }
  if (parsed >= weekStart) {
    return 'previous7days';
  }
  return 'older';
}

/** 将会话列表按日期分组（组内保持原排序，通常为 updated_at 降序）。 */
export function groupSessionsByDate(
  nodes: TreeNode[],
): Array<{ bucket: SessionDateBucket; nodes: TreeNode[] }> {
  const map = new Map<SessionDateBucket, TreeNode[]>();
  for (const bucket of SESSION_DATE_BUCKET_ORDER) {
    map.set(bucket, []);
  }
  for (const node of nodes) {
    map.get(sessionDateBucket(node.updated_at))!.push(node);
  }
  return SESSION_DATE_BUCKET_ORDER.map((bucket) => ({
    bucket,
    nodes: map.get(bucket)!,
  })).filter((group) => group.nodes.length > 0);
}

export function bumpSidebarSession(
  node: TreeNode,
  setAgents: Dispatch<SetStateAction<TreeNode[]>>,
  setPlans: Dispatch<SetStateAction<TreeNode[]>>,
  updatedAt?: string | null,
): void {
  const now = updatedAt || new Date().toISOString();
  const bump = (prev: TreeNode[]) => {
    const index = prev.findIndex((item) => item.thread_id === node.thread_id);
    if (index < 0) {
      return prev;
    }
    const item = { ...prev[index], updated_at: now };
    return [item, ...prev.slice(0, index), ...prev.slice(index + 1)];
  };
  if (node.kind === 'agent') {
    setAgents(bump);
  } else if (node.kind === 'plan') {
    setPlans(bump);
  }
}

/** 新建 Agent 尚未落盘时，保留侧栏 optimistic 条目。 */
export function prependAgentSession(
  setAgents: Dispatch<SetStateAction<TreeNode[]>>,
  node: TreeNode,
): void {
  setAgents((prev) => [node, ...prev.filter((item) => item.thread_id !== node.thread_id)]);
}
