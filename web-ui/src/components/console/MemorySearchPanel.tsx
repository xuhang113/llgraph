import { useCallback, useEffect, useState } from 'react';
import { api, type MemoryHit, type MemorySearchResult, type MemoryStatus } from '../../api/client';

interface Props {
  slug: string;
  onClose: () => void;
}

type KindFilter = 'all' | 'pref' | 'fact' | 'proc';

const KIND_LABELS: Record<string, string> = {
  pref: '偏好',
  fact: '事实',
  proc: '做法',
};

function shortId(id: string): string {
  const s = (id || '').trim();
  if (s.length <= 12) return s;
  return `${s.slice(0, 8)}…${s.slice(-4)}`;
}

function formatTime(iso: string | undefined): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso.slice(0, 19);
    return d.toLocaleString('zh-CN', { hour12: false });
  } catch {
    return iso.slice(0, 19);
  }
}

function scorePercent(score: number | undefined): number {
  if (score == null || Number.isNaN(score)) return 0;
  return Math.min(100, Math.round(score * 100));
}

function MemoryHitCard({ hit, onCopyId }: { hit: MemoryHit; onCopyId: (id: string) => void }) {
  const [expanded, setExpanded] = useState(false);
  const kind = hit.kind || 'unknown';
  const content = hit.content || '';
  const preview = content.length > 280 && !expanded ? `${content.slice(0, 280)}…` : content;
  const score = typeof hit.score === 'number' ? hit.score : undefined;

  return (
    <article className={`cursor-memory-card cursor-memory-card--${kind}`}>
      <div className="cursor-memory-card-head">
        <span className={`cursor-memory-kind cursor-memory-kind--${kind}`}>
          {KIND_LABELS[kind] || kind}
        </span>
        {hit.status && hit.status !== 'active' && (
          <span className="cursor-memory-status">{hit.status}</span>
        )}
        {score != null && (
          <div className="cursor-memory-score" title={`融合分 ${score.toFixed(4)}`}>
            <div className="cursor-memory-score-bar" style={{ width: `${scorePercent(score)}%` }} />
            <span className="cursor-memory-score-label">{score.toFixed(3)}</span>
          </div>
        )}
      </div>
      <p className="cursor-memory-content">{preview}</p>
      {content.length > 280 && (
        <button
          type="button"
          className="cursor-memory-expand"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? '收起' : `展开全文（${content.length} 字）`}
        </button>
      )}
      <div className="cursor-memory-meta">
        <span title={hit.memory_id}>
          ID{' '}
          <button type="button" className="cursor-memory-id-btn" onClick={() => onCopyId(hit.memory_id)}>
            {shortId(hit.memory_id)}
          </button>
        </span>
        <span>命中 {hit.hit_count ?? 0}</span>
        {hit.confidence != null && <span>置信 {Number(hit.confidence).toFixed(2)}</span>}
        {hit.source && <span>{hit.source}</span>}
        <span>更新 {formatTime(hit.updated_at)}</span>
        {hit.last_hit_at && <span>最近召回 {formatTime(hit.last_hit_at)}</span>}
      </div>
    </article>
  );
}

export default function MemorySearchPanel({ slug, onClose }: Props) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [minScore, setMinScore] = useState(0);
  const [kind, setKind] = useState<KindFilter>('all');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<MemorySearchResult | null>(null);
  const [status, setStatus] = useState<MemoryStatus | null>(null);
  const [copied, setCopied] = useState('');

  const loadStatus = useCallback(async () => {
    try {
      const s = await api.memoryStatus(slug);
      setStatus(s);
    } catch {
      setStatus(null);
    }
  }, [slug]);

  useEffect(() => {
    void loadStatus();
  }, [loadStatus]);

  const runSearch = async () => {
    setBusy(true);
    setError('');
    try {
      const res = await api.memorySearch(slug, query.trim(), {
        top_k: topK,
        min_score: minScore,
        kind: kind === 'all' ? undefined : kind,
      });
      setResult(res);
      if (!res.enabled) {
        setError(res.message || '长期记忆未启用');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const handleSearch = () => void runSearch();
  const handleBrowse = () => {
    setQuery('');
    void runSearch();
  };

  const handleCopyId = async (id: string) => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(id);
      window.setTimeout(() => setCopied((c) => (c === id ? '' : c)), 2000);
    } catch {
      setCopied('');
    }
  };

  const counts = status?.counts;
  const hits = result?.hits ?? [];

  return (
    <div className="cursor-modal-backdrop" onClick={onClose}>
      <div
        className="cursor-modal cursor-memory-search-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="cursor-modal-header cursor-memory-header">
          <div className="cursor-memory-title-wrap">
            <span className="cursor-memory-title-icon" aria-hidden>
              ◈
            </span>
            <div>
              <h2>长期记忆</h2>
              <p className="cursor-memory-subtitle">向量 + 关键词融合检索 · 排查 agent_memories</p>
            </div>
          </div>
          <button type="button" className="cursor-btn-ghost" onClick={onClose}>
            关闭
          </button>
        </header>

        {status && (
          <div className="cursor-memory-stats">
            <span className={`cursor-memory-pill${status.enabled ? ' is-on' : ' is-off'}`}>
              {status.enabled ? '已启用' : '未启用'}
            </span>
            {counts && (
              <>
                <span className="cursor-memory-pill">
                  active <strong>{counts.active}</strong>
                </span>
                <span className="cursor-memory-pill cursor-memory-pill--pref">
                  pref <strong>{counts.pref}</strong>
                </span>
                <span className="cursor-memory-pill cursor-memory-pill--fact">
                  fact <strong>{counts.fact}</strong>
                </span>
                <span className="cursor-memory-pill cursor-memory-pill--proc">
                  proc <strong>{counts.proc}</strong>
                </span>
              </>
            )}
            {status.memory_root && (
              <span className="cursor-memory-path" title={status.memory_root}>
                {status.memory_root}
              </span>
            )}
          </div>
        )}

        <div className="cursor-memory-toolbar">
          <div className="cursor-memory-search-row">
            <input
              className="cursor-input cursor-memory-query"
              placeholder="自然语言、关键词、偏好描述…（留空可浏览全部）"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void handleSearch()}
            />
            <input
              className="cursor-input cursor-memory-topk"
              type="number"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 20)}
              title="Top K"
            />
            <button
              type="button"
              className="cursor-btn-primary"
              disabled={busy || !status?.enabled}
              onClick={() => void handleSearch()}
            >
              {busy ? '检索中…' : '检索'}
            </button>
            <button
              type="button"
              className="cursor-btn-ghost"
              disabled={busy || !status?.enabled}
              onClick={() => void handleBrowse()}
            >
              浏览全部
            </button>
          </div>

          <div className="cursor-memory-filters">
            <span className="cursor-memory-filter-label">类型</span>
            {(['all', 'pref', 'fact', 'proc'] as KindFilter[]).map((k) => (
              <button
                key={k}
                type="button"
                className={`cursor-memory-chip${kind === k ? ' is-active' : ''} cursor-memory-chip--${k}`}
                onClick={() => setKind(k)}
              >
                {k === 'all' ? '全部' : KIND_LABELS[k]}
              </button>
            ))}
            <label className="cursor-memory-min-score">
              最低分
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.01}
                value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
              />
              <span>{minScore.toFixed(2)}</span>
            </label>
          </div>
        </div>

        {error && <div className="cursor-memory-error">{error}</div>}
        {copied && <div className="cursor-memory-toast">已复制 {shortId(copied)}</div>}

        <div className="cursor-memory-results">
          {result && (
            <div className="cursor-memory-result-meta">
              <span>
                {result.mode === 'browse' ? '浏览' : '检索'}
                {result.query ? ` · 「${result.query}」` : ''}
              </span>
              <span>
                {result.count ?? hits.length} 条
                {result.elapsed_ms != null && result.elapsed_ms > 0 && ` · ${result.elapsed_ms} ms`}
                {(result.filtered_below_min ?? 0) > 0 &&
                  ` · 过滤低分 ${result.filtered_below_min}`}
              </span>
            </div>
          )}

          {!result && !busy && (
            <div className="cursor-memory-empty">
              <span className="cursor-memory-empty-icon">◎</span>
              <p>输入查询检索记忆，或点击「浏览全部」查看当前工作区条目</p>
            </div>
          )}

          {hits.length === 0 && result && !busy && (
            <div className="cursor-memory-empty">
              <p>无匹配结果</p>
            </div>
          )}

          <div className="cursor-memory-cards">
            {hits.map((hit) => (
              <MemoryHitCard key={hit.memory_id} hit={hit} onCopyId={(id) => void handleCopyId(id)} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
