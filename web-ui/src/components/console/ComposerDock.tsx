import FileChangesPanel, { type FileChangesMode } from './FileChangesPanel';
import ChatComposer from './ChatComposer';
import type { SlashCatalogItem } from '../../api/client';

interface FileChangesProps {
  slug: string;
  mode: FileChangesMode;
  sessionThreadId: string;
  planThreadId?: string;
  taskId?: string;
  refreshKey?: number;
  onChangesUpdated?: () => void;
}

interface Props {
  fileChanges?: FileChangesProps | null;
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop?: () => void;
  busy?: boolean;
  disabled?: boolean;
  placeholder: string;
  slashCatalog?: SlashCatalogItem[];
}

export default function ComposerDock({
  fileChanges,
  value,
  onChange,
  onSend,
  onStop,
  busy = false,
  disabled = false,
  placeholder,
  slashCatalog = [],
}: Props) {
  return (
    <div className="cursor-composer-wrap">
      <div className="cursor-composer cursor-composer-dock">
        {fileChanges && (
          <FileChangesPanel
            key={`${fileChanges.mode}-${fileChanges.sessionThreadId}-${fileChanges.refreshKey ?? 0}`}
            embedded
            slug={fileChanges.slug}
            mode={fileChanges.mode}
            sessionThreadId={fileChanges.sessionThreadId}
            planThreadId={fileChanges.planThreadId}
            taskId={fileChanges.taskId}
            busy={busy}
            onStop={onStop}
            onChangesUpdated={fileChanges.onChangesUpdated}
          />
        )}
        <ChatComposer
          embedded
          value={value}
          onChange={onChange}
          onSend={onSend}
          onStop={onStop}
          busy={busy}
          disabled={disabled}
          placeholder={placeholder}
          slashCatalog={slashCatalog}
          stopInStatusBar={Boolean(fileChanges)}
        />
      </div>
    </div>
  );
}
