import type { Capabilities } from '../../api/client';
import { formatMcpToolDisplay, mcpServerZh } from '../../utils/mcpDisplay';

interface Props {
  caps: Capabilities | null;
  onTraceMode: (mode: string) => void;
}

export default function CapabilitiesPanel({ caps, onTraceMode }: Props) {
  if (!caps) {
    return <div className="caps-panel muted">加载能力清单…</div>;
  }

  return (
    <div className="caps-panel">
      <section>
        <h3>Trace</h3>
        <div className="trace-modes">
          {['all', 'steps', 'reply', 'none'].map((m) => (
            <button
              key={m}
              type="button"
              className={caps.trace_mode === m ? 'chip active' : 'chip'}
              onClick={() => onTraceMode(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </section>

      <section>
        <h3>内置工具 ({caps.builtin_tools.length})</h3>
        <ul className="caps-list">
          {caps.builtin_tools.map((t) => (
            <li key={t.name}>
              <strong>{t.name}</strong>
              <span>{t.description?.slice(0, 120)}</span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3>MCP ({caps.mcp_tools.length})</h3>
        <p className="muted small">{caps.mcp_summary}</p>
        {caps.mcp_loading && <p className="muted small">MCP 后台加载中…</p>}
        {!!caps.mcp_servers?.length && (
          <ul className="caps-list">
            {caps.mcp_servers.map((s) => (
              <li key={s.name}>
                <strong>{s.name}</strong>
                <span className="muted">
                  {' '}
                  · {mcpServerZh(s.name)} ·{' '}
                  {s.status === 'ok'
                    ? `已连接 · ${s.tool_count ?? 0} 工具`
                    : s.status === 'loading'
                      ? '加载中'
                      : s.status === 'error'
                        ? `已跳过${s.error ? `：${s.error}` : ''}`
                        : '未就绪'}
                </span>
              </li>
            ))}
          </ul>
        )}
        <ul className="caps-list">
          {caps.mcp_tools.map((t) => (
            <li key={t.name}>
              <strong>{t.name}</strong>
              <div className="small muted">{formatMcpToolDisplay(t.name, t.description)}</div>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3>Skills ({caps.skills.length})</h3>
        <ul className="caps-list">
          {caps.skills.map((s) => (
            <li key={s.name}>
              <strong>{s.name}</strong>
              <span className="muted"> [{s.scope}]</span>
              <div className="small">{s.description}</div>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3>Rules ({caps.rules.length})</h3>
        <ul className="caps-list compact">
          {caps.rules.map((r) => (
            <li key={r.id}>{r.id}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
