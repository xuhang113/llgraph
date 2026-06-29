import { useEffect, useRef, useState } from 'react';
import { resolveSessionFullTitle } from '../../utils/sessionTitleEdit';

interface Props {
  title: string;
  titleFull?: string;
  threadId: string;
  renamable?: boolean;
  onRename?: (title: string) => Promise<void>;
  className?: string;
}

export default function EditableSessionTitle({
  title,
  titleFull,
  threadId,
  renamable = false,
  onRename,
  className = 'cursor-session-title',
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const fullLabel = resolveSessionFullTitle(title, titleFull, threadId);

  useEffect(() => {
    if (!editing) {
      setDraft(fullLabel);
    }
  }, [fullLabel, editing]);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const startEdit = () => {
    if (!renamable || !onRename || saving) {
      return;
    }
    setDraft(fullLabel);
    setEditing(true);
  };

  const cancelEdit = () => {
    setDraft(fullLabel);
    setEditing(false);
  };

  const commitEdit = async () => {
    const next = draft.trim();
    if (!onRename) {
      cancelEdit();
      return;
    }
    if (!next || next === fullLabel) {
      cancelEdit();
      return;
    }
    setSaving(true);
    try {
      await onRename(next);
      setEditing(false);
    } catch {
      inputRef.current?.focus();
    } finally {
      setSaving(false);
    }
  };

  const tooltip = renamable
    ? `${fullLabel}\n${threadId}\n双击重命名`
    : `${fullLabel}\n${threadId}`;

  if (editing) {
    return (
      <div className="cursor-session-title-edit">
        <input
          ref={inputRef}
          className="cursor-session-rename-input cursor-session-title-input"
          value={draft}
          disabled={saving}
          title={fullLabel}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              void commitEdit();
            } else if (e.key === 'Escape') {
              e.preventDefault();
              cancelEdit();
            }
          }}
          onBlur={() => {
            void commitEdit();
          }}
        />
      </div>
    );
  }

  return (
    <h1
      className={`${className}${renamable ? ' cursor-session-title--editable' : ''}`}
      title={tooltip}
      onDoubleClick={renamable ? startEdit : undefined}
    >
      {title}
    </h1>
  );
}
