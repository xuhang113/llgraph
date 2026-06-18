import { useEffect, useRef, useState } from 'react';
import type { LlmModelOption } from '../../api/client';

interface Props {
  models: LlmModelOption[];
  currentId: string;
  providerLabel: string;
  disabled?: boolean;
  onSelect: (modelId: string) => void;
  switchHint?: string;
}

export default function ModelPicker({
  models,
  currentId,
  providerLabel,
  disabled = false,
  onSelect,
  switchHint = '',
}: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const current = models.find((m) => m.id === currentId) ?? models.find((m) => m.current);

  useEffect(() => {
    if (!open) {
      return;
    }
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const pick = (id: string) => {
    setOpen(false);
    if (id !== currentId) {
      onSelect(id);
    }
  };

  return (
    <div className="cursor-model-picker" ref={rootRef}>
      <button
        type="button"
        className={`cursor-model-picker-trigger${open ? ' is-open' : ''}`}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={providerLabel}
      >
        <span className="cursor-model-picker-value">{current?.id ?? currentId}</span>
        {current?.hint ? (
          <span className="cursor-model-picker-hint">{current.hint}</span>
        ) : null}
        <span className="cursor-model-picker-chevron" aria-hidden>
          ▾
        </span>
      </button>
      {switchHint ? <span className="cursor-model-picker-notice">{switchHint}</span> : null}
      {open && (
        <div className="cursor-model-picker-menu" role="listbox">
          {models.map((m) => {
            const selected = m.id === currentId;
            return (
              <button
                key={m.id}
                type="button"
                role="option"
                aria-selected={selected}
                className={`cursor-model-picker-item${selected ? ' is-selected' : ''}`}
                onClick={() => pick(m.id)}
              >
                <span className="cursor-model-picker-item-main">
                  <span className="cursor-model-picker-item-id">{m.id}</span>
                  {m.hint ? <span className="cursor-model-picker-item-desc">{m.hint}</span> : null}
                </span>
                {selected ? <span className="cursor-model-picker-check">✓</span> : null}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
