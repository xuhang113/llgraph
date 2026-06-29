import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from 'react';

export interface AppDialogAlertOptions {
  title?: string;
  message: string;
  okLabel?: string;
}

export interface AppDialogConfirmOptions {
  title?: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
}

export interface AppDialogPromptOptions {
  title?: string;
  message?: string;
  placeholder?: string;
  defaultValue?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  multiline?: boolean;
}

interface AppDialogApi {
  alert: (options: AppDialogAlertOptions | string) => Promise<void>;
  confirm: (options: AppDialogConfirmOptions | string) => Promise<boolean>;
  prompt: (options: AppDialogPromptOptions | string) => Promise<string | null>;
}

type DialogRequest =
  | {
      kind: 'alert';
      options: AppDialogAlertOptions;
      resolve: () => void;
    }
  | {
      kind: 'confirm';
      options: AppDialogConfirmOptions;
      resolve: (value: boolean) => void;
    }
  | {
      kind: 'prompt';
      options: AppDialogPromptOptions;
      resolve: (value: string | null) => void;
    };

const AppDialogContext = createContext<AppDialogApi | null>(null);

function normalizeAlert(options: AppDialogAlertOptions | string): AppDialogAlertOptions {
  return typeof options === 'string' ? { message: options } : options;
}

function normalizeConfirm(options: AppDialogConfirmOptions | string): AppDialogConfirmOptions {
  return typeof options === 'string' ? { message: options } : options;
}

function normalizePrompt(options: AppDialogPromptOptions | string): AppDialogPromptOptions {
  return typeof options === 'string' ? { message: options } : options;
}

function DialogMessage({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <div className="app-dialog-message">
      {lines.map((line, i) => (
        <p key={`${i}-${line.slice(0, 12)}`}>{line || '\u00a0'}</p>
      ))}
    </div>
  );
}

function AppDialogView({
  request,
  onClose,
}: {
  request: DialogRequest;
  onClose: () => void;
}) {
  const titleId = useId();
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
  const [promptValue, setPromptValue] = useState(
    request.kind === 'prompt' ? request.options.defaultValue || '' : '',
  );

  useEffect(() => {
    if (request.kind !== 'prompt') {
      return;
    }
    const el = inputRef.current;
    if (!el) {
      return;
    }
    el.focus();
    el.select();
  }, [request]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        if (request.kind === 'alert') {
          request.resolve();
        } else if (request.kind === 'confirm') {
          request.resolve(false);
        } else {
          request.resolve(null);
        }
        onClose();
        return;
      }
      if (e.key === 'Enter' && request.kind === 'prompt' && !request.options.multiline) {
        if ((e.target as HTMLElement).tagName === 'TEXTAREA') {
          return;
        }
        e.preventDefault();
        request.resolve(promptValue);
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [request, onClose, promptValue]);

  const dismissOverlay = () => {
    if (request.kind === 'alert') {
      request.resolve();
    } else if (request.kind === 'confirm') {
      request.resolve(false);
    } else {
      request.resolve(null);
    }
    onClose();
  };

  const title =
    request.options.title ||
    (request.kind === 'alert'
      ? '提示'
      : request.kind === 'confirm'
        ? '请确认'
        : '输入');

  const message =
    request.kind === 'prompt'
      ? request.options.message
      : request.options.message;

  return (
    <div
      className="modal-overlay app-dialog-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          dismissOverlay();
        }
      }}
    >
      <div
        className="modal app-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="app-dialog-header">
          <h2 id={titleId} className="app-dialog-title">
            {title}
          </h2>
        </header>

        {message ? <DialogMessage text={message} /> : null}

        {request.kind === 'prompt' && (
          request.options.multiline ? (
            <textarea
              ref={inputRef as React.RefObject<HTMLTextAreaElement>}
              className="app-dialog-input app-dialog-textarea"
              rows={4}
              value={promptValue}
              placeholder={request.options.placeholder}
              onChange={(e) => setPromptValue(e.target.value)}
            />
          ) : (
            <input
              ref={inputRef as React.RefObject<HTMLInputElement>}
              type="text"
              className="app-dialog-input"
              value={promptValue}
              placeholder={request.options.placeholder}
              onChange={(e) => setPromptValue(e.target.value)}
            />
          )
        )}

        <div className="modal-actions app-dialog-actions">
          {request.kind !== 'alert' && (
            <button type="button" onClick={dismissOverlay}>
              {request.kind === 'confirm'
                ? request.options.cancelLabel || '取消'
                : request.options.cancelLabel || '取消'}
            </button>
          )}
          <button
            type="button"
            className={
              request.kind === 'confirm' && request.options.danger
                ? 'app-dialog-btn-danger'
                : 'primary'
            }
            onClick={() => {
              if (request.kind === 'alert') {
                request.resolve();
              } else if (request.kind === 'confirm') {
                request.resolve(true);
              } else {
                request.resolve(promptValue);
              }
              onClose();
            }}
          >
            {request.kind === 'alert'
              ? request.options.okLabel || '确定'
              : request.kind === 'confirm'
                ? request.options.confirmLabel || '确定'
                : request.options.confirmLabel || '确定'}
          </button>
        </div>
      </div>
    </div>
  );
}

export function AppDialogProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<DialogRequest[]>([]);
  const active = queue[0] ?? null;

  const enqueue = useCallback((req: DialogRequest) => {
    setQueue((prev) => [...prev, req]);
  }, []);

  const dequeue = useCallback(() => {
    setQueue((prev) => prev.slice(1));
  }, []);

  const alert = useCallback(
    (options: AppDialogAlertOptions | string) =>
      new Promise<void>((resolve) => {
        enqueue({ kind: 'alert', options: normalizeAlert(options), resolve });
      }),
    [enqueue],
  );

  const confirm = useCallback(
    (options: AppDialogConfirmOptions | string) =>
      new Promise<boolean>((resolve) => {
        enqueue({ kind: 'confirm', options: normalizeConfirm(options), resolve });
      }),
    [enqueue],
  );

  const prompt = useCallback(
    (options: AppDialogPromptOptions | string) =>
      new Promise<string | null>((resolve) => {
        enqueue({ kind: 'prompt', options: normalizePrompt(options), resolve });
      }),
    [enqueue],
  );

  const api: AppDialogApi = { alert, confirm, prompt };

  return (
    <AppDialogContext.Provider value={api}>
      {children}
      {active && <AppDialogView request={active} onClose={dequeue} />}
    </AppDialogContext.Provider>
  );
}

export function useAppDialog(): AppDialogApi {
  const ctx = useContext(AppDialogContext);
  if (!ctx) {
    throw new Error('useAppDialog 须在 AppDialogProvider 内使用');
  }
  return ctx;
}
