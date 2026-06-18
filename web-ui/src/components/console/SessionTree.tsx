import type { TreeNode } from '../../api/client';

interface Props {
  agents: TreeNode[];
  plans: TreeNode[];
  selectedId: string | null;
  onSelect: (node: TreeNode) => void;
  onRefresh: () => void;
  onNewAgent: () => void;
  onNewPlan: () => void;
}

function NodeItem({
  node,
  selectedId,
  onSelect,
  depth,
}: {
  node: TreeNode;
  selectedId: string | null;
  onSelect: (n: TreeNode) => void;
  depth: number;
}) {
  const active = selectedId === node.thread_id;
  return (
    <>
      <button
        type="button"
        className={`tree-item kind-${node.kind} ${active ? 'active' : ''}`}
        style={{ paddingLeft: `${8 + depth * 12}px` }}
        onClick={() => onSelect(node)}
      >
        <span className="tree-kind">{node.kind}</span>
        <span className="tree-title">{node.title || node.thread_id}</span>
        {node.phase && <span className="tree-badge">{node.phase}</span>}
        {node.status && <span className="tree-badge">{node.status}</span>}
      </button>
      {node.children?.map((child) => (
        <NodeItem
          key={child.thread_id}
          node={child}
          selectedId={selectedId}
          onSelect={onSelect}
          depth={depth + 1}
        />
      ))}
    </>
  );
}

export default function SessionTree({
  agents,
  plans,
  selectedId,
  onSelect,
  onRefresh,
  onNewAgent,
  onNewPlan,
}: Props) {
  return (
    <div className="session-tree">
      <div className="tree-toolbar">
        <button type="button" onClick={onNewAgent}>
          + Agent
        </button>
        <button type="button" onClick={onNewPlan}>
          + Plan
        </button>
        <button type="button" onClick={onRefresh}>
          ↻
        </button>
      </div>
      <div className="tree-section">
        <div className="tree-section-title">Agent</div>
        {agents.map((n) => (
          <NodeItem key={n.thread_id} node={n} selectedId={selectedId} onSelect={onSelect} depth={0} />
        ))}
      </div>
      <div className="tree-section">
        <div className="tree-section-title">Plan</div>
        {plans.map((n) => (
          <NodeItem key={n.thread_id} node={n} selectedId={selectedId} onSelect={onSelect} depth={0} />
        ))}
      </div>
    </div>
  );
}
