import { useEffect, useState } from 'react';
import ChatRoleAvatar from './ChatRoleAvatar';
import ChatUserQueryBlock from './ChatUserQueryBlock';
import type { ChatImageAttachment } from '../../types/chatImage';

interface Props {
  visible: boolean;
  messageId: string | null;
  text: string;
  images?: ChatImageAttachment[];
  onJump: () => void;
}

export default function ChatFloatingUserPrompt({
  visible,
  messageId,
  text,
  images,
  onJump,
}: Props) {
  const trimmed = text.trim();
  const hasContent = trimmed.length > 0 || (images?.length ?? 0) > 0;
  const [displayText, setDisplayText] = useState(trimmed);
  const [textFading, setTextFading] = useState(false);

  useEffect(() => {
    if (trimmed === displayText) {
      return;
    }
    if (!visible) {
      setDisplayText(trimmed);
      return;
    }
    setTextFading(true);
    const timer = window.setTimeout(() => {
      setDisplayText(trimmed);
      setTextFading(false);
    }, 90);
    return () => window.clearTimeout(timer);
  }, [trimmed, displayText, visible]);

  if (!hasContent && !messageId) {
    return null;
  }

  return (
    <div
      className={`cursor-floating-user-prompt${visible ? ' is-visible' : ' is-hidden'}`}
      role="region"
      aria-label="当前问题"
      aria-hidden={!visible}
    >
      <article className="cursor-msg cursor-msg--user cursor-msg--float">
        <div className="cursor-msg-row">
          <button
            type="button"
            className="cursor-msg-role-jump"
            onClick={onJump}
            title="跳转到问题"
            aria-label="跳转到问题"
          >
            <ChatRoleAvatar role="user" />
          </button>
          <div className="cursor-msg-body">
            <div className={textFading ? 'is-text-fading' : undefined}>
              <ChatUserQueryBlock
                text={displayText}
                images={images}
                collapsed
                expandable
                expandedMaxHeight="min(36vh, 280px)"
              />
            </div>
          </div>
        </div>
      </article>
    </div>
  );
}
