import { useEffect, useState } from 'react';
import { api, type Capabilities } from '../../api/client';
import MarkdownView from './MarkdownView';
import { useAppDialog } from '../appDialogContext';
import { formatMcpToolDisplay, mcpServerZh } from '../../utils/mcpDisplay';

type CatalogKind = 'skills' | 'rules' | 'tools';

interface Props {
  slug: string;
  kind: CatalogKind;
  caps: Capabilities | null;
  onClose: () => void;
  onCapsRefresh?: () => void;
}

function scopeText(scope: string, scopeLabel?: string): string {
  return scopeLabel || (scope === 'user' ? '个人' : scope === 'workspace' ? '项目' : scope);
}

function ruleBasename(ruleId: string): string {
  const slash = ruleId.lastIndexOf('/');
  return slash >= 0 ? ruleId.slice(slash + 1) : ruleId;
}

interface DetailState {
  title: string;
  path: string;
  scope: string;
  body: string;
}

export default function CatalogPanel({ slug, kind, caps, onClose, onCapsRefresh }: Props) {
  const { alert } = useAppDialog();
  const [detail, setDetail] = useState<DetailState | null>(null);
  const [loading, setLoading] = useState(false);
  const [toggleBusy, setToggleBusy] = useState<string | null>(null);
  const [toolsTab, setToolsTab] = useState<'builtin' | 'mcp'>('builtin');

  useEffect(() => {
    setDetail(null);
  }, [kind, slug]);

  useEffect(() => {
    if (kind === 'tools') {
      setToolsTab('builtin');
    }
  }, [kind, slug]);

  const titleMap: Record<CatalogKind, string> = {
    skills: 'Skills',
    rules: 'Rules',
    tools: '工具',
  };

  const loadSkill = async (name: string) => {
    setLoading(true);
    try {
      const d = await api.skillDetail(slug, name);
      setDetail({
        title: d.name,
        path: d.path,
        scope: scopeText(d.scope, d.scope_label),
        body: d.body,
      });
    } catch (e) {
      setDetail({
        title: name,
        path: '',
        scope: '',
        body: String(e),
      });
    } finally {
      setLoading(false);
    }
  };

  const loadRule = async (id: string) => {
    setLoading(true);
    try {
      const d = await api.ruleDetail(slug, id);
      setDetail({
        title: d.id,
        path: d.path,
        scope: scopeText(d.scope, d.scope_label),
        body: d.body,
      });
    } catch (e) {
      setDetail({
        title: id,
        path: '',
        scope: '',
        body: String(e),
      });
    } finally {
      setLoading(false);
    }
  };

  const builtinTools = caps?.builtin_tools ?? [];
  const mcpTools = caps?.mcp_tools ?? [];
  const mcpServers = caps?.mcp_servers ?? [];

  const handleSkillToggle = async (name: string, active: boolean) => {
    setToggleBusy(name);
    try {
      await api.toggleSkill(slug, name, active);
      onCapsRefresh?.();
    } catch (e) {
      await alert(e instanceof Error ? e.message : String(e));
    } finally {
      setToggleBusy(null);
    }
  };

  const handleRuleToggle = async (id: string, enabled: boolean) => {
    setToggleBusy(id);
    try {
      await api.toggleRule(slug, id, enabled);
      onCapsRefresh?.();
    } catch (e) {
      await alert(e instanceof Error ? e.message : String(e));
    } finally {
      setToggleBusy(null);
    }
  };

  const ruleEnabled = (r: Capabilities['rules'][number]) => {
    if (r.disabled) {
      return false;
    }
    if (r.forced) {
      return true;
    }
    return null;
  };

  return (
    <div className="cursor-catalog">
      <header className="cursor-catalog-header">
        <h2>{titleMap[kind]}</h2>
        <button type="button" className="cursor-btn-ghost" onClick={onClose}>
          返回会话
        </button>
      </header>
      {kind === 'tools' && (
        <div className="cursor-catalog-tools-tabs" role="tablist" aria-label="工具分类">
          <button
            type="button"
            role="tab"
            aria-selected={toolsTab === 'builtin'}
            className={`cursor-catalog-tools-tab${toolsTab === 'builtin' ? ' is-active' : ''}`}
            onClick={() => setToolsTab('builtin')}
          >
            内置工具 ({builtinTools.length})
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={toolsTab === 'mcp'}
            className={`cursor-catalog-tools-tab${toolsTab === 'mcp' ? ' is-active' : ''}`}
            onClick={() => setToolsTab('mcp')}
          >
            MCP ({mcpTools.length})
          </button>
        </div>
      )}
      <div className={`cursor-catalog-body${kind === 'tools' ? ' cursor-catalog-body--single' : ''}`}>
        <div className="cursor-catalog-list">
          {kind === 'skills' &&
            (caps?.skills.length ? (
              caps.skills.map((s) => (
                <div
                  key={s.name}
                  className={`cursor-catalog-item-row${detail?.title === s.name ? ' is-active' : ''}`}
                >
                  <button
                    type="button"
                    className={`cursor-catalog-pin${s.active ? ' is-active' : ''}`}
                    title={s.active ? '取消置顶' : '置顶技能'}
                    disabled={toggleBusy === s.name}
                    onClick={(e) => {
                      e.stopPropagation();
                      void handleSkillToggle(s.name, !s.active);
                    }}
                  >
                    {s.active ? '★' : '☆'}
                  </button>
                  <button
                    type="button"
                    className="cursor-catalog-item cursor-catalog-item--flex"
                    onClick={() => loadSkill(s.name)}
                  >
                    <span className="cursor-catalog-item-title">{s.name}</span>
                    <span className="cursor-catalog-item-desc">{s.description}</span>
                    <span className="cursor-catalog-item-meta">{scopeText(s.scope, s.scope_label)}</span>
                  </button>
                </div>
              ))
            ) : (
              <div className="cursor-catalog-empty">暂无 Skill</div>
            ))}
          {kind === 'rules' &&
            (caps?.rules.length ? (
              caps.rules.map((r) => {
                const enabled = ruleEnabled(r);
                return (
                  <div
                    key={r.id}
                    className={`cursor-catalog-item-row${detail?.title === r.id ? ' is-active' : ''}`}
                  >
                    <button
                      type="button"
                      className={`cursor-catalog-rule-toggle${enabled === true ? ' is-on' : enabled === false ? ' is-off' : ''}`}
                      title={
                        enabled === true
                          ? '已强制启用，点击禁用'
                          : enabled === false
                            ? '已禁用，点击启用'
                            : '默认匹配，点击强制启用'
                      }
                      disabled={toggleBusy === r.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleRuleToggle(r.id, enabled !== true);
                      }}
                    >
                      {enabled === true ? 'ON' : enabled === false ? 'OFF' : '—'}
                    </button>
                    <button
                      type="button"
                      className="cursor-catalog-item cursor-catalog-item--flex"
                      onClick={() => loadRule(r.id)}
                    >
                      <span className="cursor-catalog-item-title">{ruleBasename(r.id)}</span>
                      <span className="cursor-catalog-item-desc">{r.description}</span>
                      <span className="cursor-catalog-item-meta">
                        {scopeText(r.scope, r.scope_label)}
                        <span className="cursor-catalog-item-id">{r.id}</span>
                      </span>
                    </button>
                  </div>
                );
              })
            ) : (
              <div className="cursor-catalog-empty">暂无 Rule</div>
            ))}
          {kind === 'tools' && toolsTab === 'builtin' &&
            (builtinTools.length ? (
              builtinTools.map((t) => (
                <div key={t.name} className="cursor-catalog-item cursor-catalog-item--static">
                  <span className="cursor-catalog-item-title">{t.name}</span>
                  <span className="cursor-catalog-item-desc">{t.description}</span>
                </div>
              ))
            ) : (
              <div className="cursor-catalog-empty">暂无内置工具</div>
            ))}
          {kind === 'tools' && toolsTab === 'mcp' && (
            <>
              {caps?.mcp_loading && (
                <div className="cursor-catalog-empty">MCP 后台加载中…</div>
              )}
              {!!mcpServers.length && (
                <div className="cursor-catalog-item cursor-catalog-item--static">
                  <span className="cursor-catalog-item-title">MCP Servers</span>
                  <span className="cursor-catalog-item-desc">
                    {mcpServers
                      .map((s) => {
                        const st =
                          s.status === 'ok'
                            ? `已连接 · ${s.tool_count ?? 0} 工具`
                            : s.status === 'loading'
                              ? '加载中'
                              : s.status === 'error'
                                ? `已跳过${s.error ? `：${s.error}` : ''}`
                                : '未就绪';
                        return `${s.name}：${mcpServerZh(s.name)}（${st}）`;
                      })
                      .join('\n')}
                  </span>
                </div>
              )}
              {!!caps?.mcp_summary && (
                <div className="cursor-catalog-item cursor-catalog-item--static">
                  <span className="cursor-catalog-item-title">摘要</span>
                  <span className="cursor-catalog-item-desc">{caps.mcp_summary}</span>
                </div>
              )}
              {mcpTools.length ? (
                mcpTools.map((t) => (
                  <div key={t.name} className="cursor-catalog-item cursor-catalog-item--static">
                    <span className="cursor-catalog-item-title">{t.name}</span>
                    <span className="cursor-catalog-item-desc">
                      {formatMcpToolDisplay(t.name, t.description)}
                    </span>
                  </div>
                ))
              ) : (
                !caps?.mcp_loading && (
                  <div className="cursor-catalog-empty">暂无可用 MCP 工具</div>
                )
              )}
            </>
          )}
        </div>
        {kind !== 'tools' && (
          <div className="cursor-catalog-detail">
            {loading && <div className="cursor-catalog-empty">加载中…</div>}
            {!loading && !detail && (
              <div className="cursor-catalog-empty">点击左侧条目查看详情</div>
            )}
            {!loading && detail && (
              <>
                <h3>{detail.title}</h3>
                {detail.path && (
                  <div className="cursor-catalog-path" title={detail.path}>{detail.path}</div>
                )}
                {detail.scope && (
                  <div className="cursor-catalog-scope">来源: {detail.scope}</div>
                )}
                <div className="cursor-catalog-markdown">
                  <MarkdownView content={detail.body} />
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
