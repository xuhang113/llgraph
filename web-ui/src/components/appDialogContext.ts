import { createContext, useContext } from 'react';

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

export interface AppDialogApi {
  alert: (options: AppDialogAlertOptions | string) => Promise<void>;
  confirm: (options: AppDialogConfirmOptions | string) => Promise<boolean>;
  prompt: (options: AppDialogPromptOptions | string) => Promise<string | null>;
}

export const AppDialogContext = createContext<AppDialogApi | null>(null);

/** 与 AppDialogProvider 分文件，避免 Vite Fast Refresh 因混合导出失效导致黑屏。 */
export function useAppDialog(): AppDialogApi {
  const ctx = useContext(AppDialogContext);
  if (!ctx) {
    throw new Error('useAppDialog 须在 AppDialogProvider 内使用');
  }
  return ctx;
}
