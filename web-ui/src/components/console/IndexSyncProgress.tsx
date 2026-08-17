import type { IndexProgress } from '../../api/client';

function actionLabel(action?: string | null): string {
  switch (action) {
    case 'incremental':
      return '增量';
    case 'full':
      return '全量';
    case 'rebuild':
      return '重建';
    case 'dry-run':
      return '演练';
    case 'watch':
      return 'Watch';
    default:
      return '同步';
  }
}

/** 与后端 estimate_progress_percent 对齐：以扫描为主，避免卡在虚假高百分比。 */
export function estimateIndexPercent(progress: IndexProgress): number | null {
  if (!progress.running && progress.ok !== false) {
    return 100;
  }
  const total = progress.files_total;
  const scanned = progress.files_scanned ?? 0;
  const skipped = progress.files_skipped ?? 0;
  const updated = progress.files_updated ?? 0;
  const pending = Math.max(0, scanned - skipped);
  if (total == null || total <= 0) {
    if (pending > 0 && updated > 0) {
      return Math.min(99, (updated / pending) * 100);
    }
    return scanned > 0 ? null : 0;
  }
  const scanPct = Math.min(99, (scanned / total) * 100);
  if (pending <= 0) {
    return scanPct;
  }
  if (updated <= 0) {
    return Math.min(99, scanPct * 0.92);
  }
  const embedRatio = Math.min(1, updated / pending);
  if (embedRatio >= 0.95) {
    return scanPct;
  }
  return Math.min(99, scanPct * (0.7 + 0.3 * embedRatio));
}

export function indexProgressDetail(progress: IndexProgress): string {
  if (progress.phase === 'prepare') {
    return '准备中…';
  }
  if (progress.phase === 'done') {
    return progress.ok === false && progress.error
      ? `结束：${progress.error}`
      : '已完成';
  }
  const scanned = progress.files_scanned ?? 0;
  const skipped = progress.files_skipped ?? 0;
  const updated = progress.files_updated ?? 0;
  const chunks = progress.chunks_written ?? 0;
  const total = progress.files_total;
  const scanPart =
    total != null && total > 0
      ? `扫描 ${scanned.toLocaleString()}/${total.toLocaleString()}`
      : `扫描 ${scanned.toLocaleString()}`;
  return `${scanPart}（跳过 ${skipped.toLocaleString()}）· 索引 ${updated.toLocaleString()} / ${chunks.toLocaleString()} chunks`;
}

/** 索引同步进度条（确定百分比或不确定动画）。 */
export default function IndexSyncProgress({
  progress,
  compact = false,
}: {
  progress: IndexProgress;
  compact?: boolean;
}) {
  const estimated = estimateIndexPercent(progress);
  const pct =
    estimated != null && Number.isFinite(estimated)
      ? Math.max(0, Math.min(100, estimated))
      : typeof progress.percent === 'number' && Number.isFinite(progress.percent)
        ? Math.max(0, Math.min(100, progress.percent))
        : null;
  const determinate = pct !== null && (progress.running || pct >= 100);
  const title = `${actionLabel(progress.action)} · ${indexProgressDetail(progress)}`;
  const elapsed =
    typeof progress.elapsed_sec === 'number' && progress.elapsed_sec > 0
      ? ` · ${progress.elapsed_sec.toFixed(0)}s`
      : '';

  return (
    <div className={`cursor-index-progress${compact ? ' is-compact' : ''}`}>
      <div className="cursor-index-progress-head">
        <span className="cursor-index-progress-title">{title}</span>
        <span className="cursor-index-progress-meta">
          {determinate ? `${pct.toFixed(0)}%` : progress.running ? '进行中' : ''}
          {elapsed}
        </span>
      </div>
      <div
        className={`cursor-index-progress-track${determinate ? '' : ' is-indeterminate'}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={determinate ? Math.round(pct) : undefined}
        aria-busy={progress.running || undefined}
      >
        {determinate ? (
          <div className="cursor-index-progress-fill" style={{ width: `${pct}%` }} />
        ) : (
          <div className="cursor-index-progress-fill is-pulse" />
        )}
      </div>
    </div>
  );
}
