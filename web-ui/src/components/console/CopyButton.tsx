import { useState } from 'react';
import { copyTextToClipboard } from '../../utils/clipboard';

interface Props {
  text: string;
  title?: string;
  className?: string;
  iconOnly?: boolean;
}

function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"
      />
    </svg>
  );
}

export default function CopyButton({
  text,
  title = '复制',
  className = '',
  iconOnly = false,
}: Props) {
  const [copied, setCopied] = useState(false);

  const handleClick = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const ok = await copyTextToClipboard(text);
    if (!ok) {
      return;
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  const label = copied ? '已复制' : title;

  return (
    <button
      type="button"
      className={`cursor-copy-btn${iconOnly ? ' cursor-copy-btn--icon' : ''}${copied ? ' is-copied' : ''} ${className}`.trim()}
      onClick={(e) => void handleClick(e)}
      title={label}
      aria-label={label}
    >
      <CopyIcon />
      {!iconOnly && <span>{label}</span>}
    </button>
  );
}
