import type { Dispatch, SetStateAction } from 'react';
import type { TreeNode } from '../../api/client';

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
