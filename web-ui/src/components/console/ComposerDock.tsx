import FileChangesPanel, { type FileChangesMode } from './FileChangesPanel';
import ChatComposer from './ChatComposer';
import type { SlashCatalogItem } from '../../api/client';
import type { ComposerImage } from '../../types/chatImage';

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
  images: ComposerImage[];
  onImagesChange: (images: ComposerImage[]) => void;
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
  images,
  onImagesChange,
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
          images={images}
          onImagesChange={onImagesChange}
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
