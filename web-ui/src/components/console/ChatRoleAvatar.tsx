interface Props {
  role: 'user' | 'assistant';
}

const LABELS = { user: '你', assistant: '助手' } as const;

export default function ChatRoleAvatar({ role }: Props) {
  return (
    <div className={`cursor-msg-role-col cursor-msg-role-col--${role}`}>
      <div className={`cursor-msg-avatar cursor-msg-avatar--${role}`} aria-hidden="true">
        {role === 'user' ? (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="5.5" r="2.5" stroke="currentColor" strokeWidth="1.4" />
            <path
              d="M3.5 13.5c0-2.5 2-4 4.5-4s4.5 1.5 4.5 4"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M8 2.5l1.1 2.4 2.6.3-1.9 1.7.6 2.6L8 8.4 5.6 9.5l.6-2.6-1.9-1.7 2.6-.3L8 2.5z"
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinejoin="round"
            />
            <path
              d="M4 12.5h8"
              stroke="currentColor"
              strokeWidth="1.3"
              strokeLinecap="round"
            />
          </svg>
        )}
      </div>
      <span className="cursor-msg-role-name">{LABELS[role]}</span>
    </div>
  );
}
