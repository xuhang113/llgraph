import type {
  ContextBreakdownSection,
  ContextDetail,
  ContextMessageInspect,
} from '../../api/client';
import { formatContextTokens } from '../../utils/contextDisplay';

const MESSAGE_KIND_LABELS: Record<string, string> = {
  manifest: 'Manifest',
  anchor: 'Anchor',
  summary: 'Summary',
  user: 'User',
  assistant: 'Assistant',
  tool: 'Tool',
  system: 'System',
};

interface Props {
  detail: ContextDetail | null;
  loading?: boolean;
  compact?: boolean;
}

function MessageInspectList({
  messages,
  emptyText,
}: {
  messages: ContextMessageInspect[];
  emptyText: string;
}) {
  if (messages.length === 0) {
    return <p className="muted small">{emptyText}</p>;
  }
  return (
    <div className="cursor-context-message-list">
      {messages.map((msg) => (
        <details key={`${msg.index}-${msg.kind}-${msg.role}`} className="cursor-context-message-item">
          <summary className="cursor-context-message-summary">
            <span className="cursor-context-message-index">#{msg.index}</span>
            <span className="cursor-context-message-kind">
              {MESSAGE_KIND_LABELS[msg.kind] || msg.kind}
            </span>
            {msg.tool_name && (
              <span className="cursor-context-message-tool">{msg.tool_name}</span>
            )}
            <span className="cursor-context-message-meta">
              ~{formatContextTokens(msg.tokens)}
            </span>
          </summary>
          <pre className="cursor-context-message-body">{msg.preview}</pre>
        </details>
      ))}
    </div>
  );
}

function BreakdownSectionItem({ section }: { section: ContextBreakdownSection }) {
  const messages = section.messages ?? [];
  const hasMessages = messages.length > 0;
  const hasPreview = Boolean(section.preview.trim());
  const emptyHint = section.empty_hint?.trim() ?? '';

  return (
    <details className="cursor-context-breakdown-item">
      <summary className="cursor-context-breakdown-summary">
        <span className="cursor-context-breakdown-title">{section.title}</span>
        <span className="cursor-context-breakdown-meta">
          ~{formatContextTokens(section.tokens)}
        </span>
      </summary>
      <div className="cursor-context-breakdown-body">
        {emptyHint && !hasMessages && !hasPreview && (
          <p className="muted small">{emptyHint}</p>
        )}
        {section.from_disk && hasPreview && (
          <p className="muted small">来自 conversation_anchor.json（落盘摘要）</p>
        )}
        {hasMessages ? (
          <MessageInspectList messages={messages} emptyText="暂无消息" />
        ) : hasPreview ? (
          <pre className="cursor-context-message-body">{section.preview}</pre>
        ) : !emptyHint ? (
          <p className="muted small">暂无预览</p>
        ) : null}
      </div>
    </details>
  );
}

export default function ContextDetailSections({ detail, loading, compact = false }: Props) {
  if (loading && !detail) {
    return <p className="muted small">加载上下文详情…</p>;
  }
  if (!detail) {
    return null;
  }

  const sections =
    detail.breakdown_sections.length > 0
      ? detail.breakdown_sections
      : detail.fixed_sections.map((section) => ({
          key: section.key,
          title: section.title,
          tokens: section.tokens,
          preview: section.preview,
          truncated: section.truncated,
        }));

  if (sections.length === 0) {
    return null;
  }

  return (
    <div className={`cursor-context-detail${compact ? ' is-compact' : ''}`}>
      <section className="cursor-context-section">
        <h4 className="cursor-context-detail-title">Token 分项</h4>
        <div className="cursor-context-breakdown-list">
          {sections.map((section) => (
            <BreakdownSectionItem key={section.key} section={section} />
          ))}
        </div>
      </section>
    </div>
  );
}
