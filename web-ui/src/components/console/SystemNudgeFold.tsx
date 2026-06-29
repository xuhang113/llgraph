interface Props {
  summary?: string;
  detail: string;
}

/** Agent 内部续跑提示（think_nudge），默认折叠，不占用「你」消息位。 */
export default function SystemNudgeFold({ summary, detail }: Props) {
  const label = summary?.trim() || '系统续跑';
  const body = detail.trim();
  if (!body) {
    return null;
  }

  return (
    <details className="cursor-system-nudge-fold">
      <summary className="cursor-system-nudge-fold-summary">
        <span className="cursor-system-nudge-fold-icon" aria-hidden>
          ⚙
        </span>
        <span className="cursor-system-nudge-fold-label">{label}</span>
        <span className="cursor-system-nudge-fold-hint">点击展开</span>
      </summary>
      <div className="cursor-system-nudge-fold-body">{body}</div>
    </details>
  );
}
