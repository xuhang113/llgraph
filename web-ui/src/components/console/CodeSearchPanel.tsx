import { useState } from 'react';
import { api } from '../../api/client';

interface Props {
  slug: string;
  onClose: () => void;
}

export default function CodeSearchPanel({ slug, onClose }: Props) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'parallel' | 'semantic'>('parallel');
  const [topK, setTopK] = useState(15);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!query.trim()) {
      return;
    }
    setBusy(true);
    setError('');
    try {
      const res = await api.codeSearch(slug, query.trim(), { mode, top_k: topK });
      setResult(res.text || '（无结果）');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult('');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="cursor-modal-backdrop" onClick={onClose}>
      <div
        className="cursor-modal cursor-code-search-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="cursor-modal-header">
          <h2>搜代码</h2>
          <button type="button" className="cursor-btn-ghost" onClick={onClose}>
            关闭
          </button>
        </header>
        <div className="cursor-code-search-form">
          <input
            className="cursor-input"
            placeholder="类名、关键字、自然语言问题…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void handleSearch()}
          />
          <select
            className="cursor-input cursor-code-search-select"
            value={mode}
            onChange={(e) => setMode(e.target.value as 'parallel' | 'semantic')}
          >
            <option value="parallel">并行（grep + 向量）</option>
            <option value="semantic">纯向量</option>
          </select>
          <input
            className="cursor-input cursor-code-search-topk"
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value) || 15)}
            title="Top K"
          />
          <button
            type="button"
            className="cursor-btn-primary"
            disabled={busy || !query.trim()}
            onClick={() => void handleSearch()}
          >
            {busy ? '搜索中…' : '搜索'}
          </button>
        </div>
        {error && <div className="cursor-code-search-error">{error}</div>}
        <pre className="cursor-code-search-result">{result || '输入查询后搜索'}</pre>
      </div>
    </div>
  );
}
