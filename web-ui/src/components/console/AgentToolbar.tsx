import { useEffect, useRef, useState } from 'react';
import type { LlmSettings } from '../../api/client';
import ModelPicker from './ModelPicker';

interface Props {
  llm: LlmSettings | null;
  busy: boolean;
  isAgent: boolean;
  webSearchEnabled?: boolean;
  onModelChange: (modelId: string) => Promise<void>;
  onThinkingChange: (enabled: boolean) => void;
  onWebSearchChange?: (enabled: boolean) => Promise<void>;
}

export default function AgentToolbar({
  llm,
  busy,
  isAgent,
  webSearchEnabled = false,
  onModelChange,
  onThinkingChange,
  onWebSearchChange,
}: Props) {
  const [switchHint, setSwitchHint] = useState('');
  const [webHint, setWebHint] = useState('');
  const hintTimerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (hintTimerRef.current !== null) {
        window.clearTimeout(hintTimerRef.current);
      }
    },
    [],
  );

  if (!isAgent) {
    return null;
  }

  const handleModelPick = async (modelId: string) => {
    try {
      await onModelChange(modelId);
      setSwitchHint('下一条消息起生效');
      if (hintTimerRef.current !== null) {
        window.clearTimeout(hintTimerRef.current);
      }
      hintTimerRef.current = window.setTimeout(() => setSwitchHint(''), 2800);
    } catch (err) {
      setSwitchHint(err instanceof Error ? err.message : '切换失败');
      hintTimerRef.current = window.setTimeout(() => setSwitchHint(''), 3200);
    }
  };

  return (
    <div className="cursor-agent-toolbar">
      {llm && llm.models.length > 0 && (
        <div className="cursor-agent-toolbar-field">
          <span className="cursor-agent-toolbar-label">模型</span>
          <ModelPicker
            models={llm.models}
            currentId={llm.model}
            providerLabel={llm.provider_label}
            disabled={busy}
            switchHint={switchHint}
            onSelect={handleModelPick}
          />
        </div>
      )}

      {llm?.thinking.supported && (
        <label className="cursor-agent-toolbar-toggle" title="扩展思考（thinking）">
          <input
            type="checkbox"
            checked={llm.thinking.enabled}
            disabled={busy}
            onChange={(e) => onThinkingChange(e.target.checked)}
          />
          Thinking
        </label>
      )}

      {onWebSearchChange && (
        <label className="cursor-agent-toolbar-toggle" title="联网搜索（Tavily web_search 工具）">
          <input
            type="checkbox"
            checked={webSearchEnabled}
            disabled={busy}
            onChange={async (e) => {
              try {
                await onWebSearchChange(e.target.checked);
                setWebHint(e.target.checked ? '已启用' : '已禁用');
                window.setTimeout(() => setWebHint(''), 2400);
              } catch (err) {
                setWebHint(err instanceof Error ? err.message : '切换失败');
                window.setTimeout(() => setWebHint(''), 3200);
              }
            }}
          />
          联网
          {webHint && <span className="cursor-agent-toolbar-hint">{webHint}</span>}
        </label>
      )}
    </div>
  );
}
