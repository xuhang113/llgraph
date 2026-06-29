import type { Workspace } from '../api/client';

const SLUG_KEY = 'llgraph.lastWorkspaceSlug';
const META_KEY = 'llgraph.lastWorkspaceMeta';
const RECENT_KEY = 'llgraph.recentWorkspaces';

export interface StoredWorkspaceMeta {
  slug: string;
  path: string;
  label: string;
}

export function readStoredWorkspaceSlug(): string {
  try {
    return localStorage.getItem(SLUG_KEY) || '';
  } catch {
    return '';
  }
}

export function readStoredWorkspaceMeta(): StoredWorkspaceMeta {
  const slug = readStoredWorkspaceSlug();
  try {
    const raw = localStorage.getItem(META_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<StoredWorkspaceMeta>;
      if (parsed && typeof parsed.slug === 'string') {
        return {
          slug: slug || parsed.slug,
          path: typeof parsed.path === 'string' ? parsed.path : '',
          label: typeof parsed.label === 'string' ? parsed.label : '',
        };
      }
    }
  } catch {
    /* ignore */
  }
  return { slug, path: '', label: '' };
}

export function writeStoredWorkspaceMeta(meta: StoredWorkspaceMeta): void {
  if (!meta.slug) {
    return;
  }
  try {
    localStorage.setItem(SLUG_KEY, meta.slug);
    localStorage.setItem(
      META_KEY,
      JSON.stringify({
        slug: meta.slug,
        path: meta.path,
        label: meta.label,
      }),
    );
  } catch {
    /* ignore */
  }
}

function isWorkspaceRecord(value: unknown): value is Workspace {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const row = value as Record<string, unknown>;
  return typeof row.slug === 'string' && typeof row.path === 'string';
}

/** 上次成功拉取的最近工作区列表（刷新时 API 未就绪时兜底展示）。 */
export function readStoredRecentWorkspaces(): Workspace[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isWorkspaceRecord).map((w) => ({
      slug: w.slug,
      path: w.path,
      session_count: typeof w.session_count === 'number' ? w.session_count : 0,
      plan_count: typeof w.plan_count === 'number' ? w.plan_count : 0,
      updated_at: typeof w.updated_at === 'string' ? w.updated_at : null,
    }));
  } catch {
    return [];
  }
}

export function writeStoredRecentWorkspaces(workspaces: Workspace[]): void {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(workspaces));
  } catch {
    /* ignore */
  }
}

export function isPackagedExampleWorkspacePath(path: string): boolean {
  const normalized = (path || '').replace(/\\/g, '/').toLowerCase();
  return (
    normalized.endsWith('/examples/user-llgraph') ||
    normalized.endsWith('/examples/default-workspace')
  );
}

export function mergeWorkspaceCatalog(
  apiList: Workspace[],
  previous: Workspace[],
  pinnedSlug: string,
): Workspace[] {
  const bySlug = new Map<string, Workspace>();
  for (const w of previous) {
    bySlug.set(w.slug, w);
  }
  for (const w of apiList) {
    bySlug.set(w.slug, w);
  }
  if (pinnedSlug && !bySlug.has(pinnedSlug)) {
    const fromPrev = previous.find((w) => w.slug === pinnedSlug);
    const fromRecent = readStoredRecentWorkspaces().find((w) => w.slug === pinnedSlug);
    const pick = fromPrev ?? fromRecent;
    if (pick) {
      bySlug.set(pinnedSlug, pick);
    } else {
      const meta = readStoredWorkspaceMeta();
      if (meta.slug === pinnedSlug && meta.path) {
        bySlug.set(pinnedSlug, {
          slug: pinnedSlug,
          path: meta.path,
          session_count: 0,
          plan_count: 0,
          updated_at: null,
        });
      }
    }
  }
  return [...bySlug.values()]
    .filter((w) => !isPackagedExampleWorkspacePath(w.path))
    .sort((a, b) => {
    const ta = a.updated_at ?? '';
    const tb = b.updated_at ?? '';
    return tb.localeCompare(ta);
  });
}

export function workspaceLabelFromPath(path: string, slug: string): string {
  const trimmed = (path || '').trim();
  if (!trimmed) {
    return slug;
  }
  const parts = trimmed.split('/').filter(Boolean);
  return parts[parts.length - 1] || slug;
}
