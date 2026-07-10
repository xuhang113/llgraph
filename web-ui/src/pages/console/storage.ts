import type { LlmSettings, Workspace } from '../../api/client';
import { readStoredRecentWorkspaces, readStoredWorkspaceSlug } from '../../utils/workspaceStorage';
import {
  LAST_SESSION_THREAD_KEY,
  RIGHT_PANEL_WIDTH_DEFAULT,
  RIGHT_PANEL_WIDTH_MAX,
  RIGHT_PANEL_WIDTH_MIN,
  SIDEBAR_WIDTH_DEFAULT,
  SIDEBAR_WIDTH_MAX,
  SIDEBAR_WIDTH_MIN,
} from './constants';

export function readStoredPanelWidth(key: string, fallback: number, min: number, max: number): number {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) {
      return fallback;
    }
    const n = Number.parseInt(raw, 10);
    if (!Number.isFinite(n)) {
      return fallback;
    }
    return Math.min(max, Math.max(min, n));
  } catch {
    return fallback;
  }
}

export function clampPanelWidth(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.round(value)));
}

export function readStoredSessionThread(slug: string): string {
  if (!slug) {
    return '';
  }
  try {
    return localStorage.getItem(`${LAST_SESSION_THREAD_KEY}:${slug}`) || '';
  } catch {
    return '';
  }
}

export function writeStoredSessionThread(slug: string, threadId: string) {
  if (!slug || !threadId) {
    return;
  }
  try {
    localStorage.setItem(`${LAST_SESSION_THREAD_KEY}:${slug}`, threadId);
  } catch {
    /* ignore */
  }
}

export function resolveWorkspaceSlug(current: string, workspaces: Workspace[]): string {
  if (workspaces.length === 0) {
    return current;
  }
  if (current && workspaces.some((w) => w.slug === current)) {
    return current;
  }
  const saved = readStoredWorkspaceSlug();
  if (saved && workspaces.some((w) => w.slug === saved)) {
    return saved;
  }
  const cached = readStoredRecentWorkspaces();
  if (current && cached.some((w) => w.slug === current)) {
    return current;
  }
  if (saved && cached.some((w) => w.slug === saved)) {
    return saved;
  }
  return workspaces[0].slug;
}

export function readInitialSidebarWidth(): number {
  return readStoredPanelWidth(
    'llgraph-sidebar-width',
    SIDEBAR_WIDTH_DEFAULT,
    SIDEBAR_WIDTH_MIN,
    SIDEBAR_WIDTH_MAX,
  );
}

export function readInitialRightPanelWidth(): number {
  return readStoredPanelWidth(
    'llgraph-right-panel-width',
    RIGHT_PANEL_WIDTH_DEFAULT,
    RIGHT_PANEL_WIDTH_MIN,
    RIGHT_PANEL_WIDTH_MAX,
  );
}

const ALLOW_WRITE_KEY = 'llgraph-allow-write';
const ALLOW_WRITE_VERSION_KEY = 'llgraph-allow-write-version';
const ALLOW_WRITE_VERSION = '3';

/** 工具栏「允许写」偏好；Web 缺省为 true。 */
export function readStoredAllowWrite(): boolean {
  try {
    const version = localStorage.getItem(ALLOW_WRITE_VERSION_KEY);
    if (version !== ALLOW_WRITE_VERSION) {
      localStorage.setItem(ALLOW_WRITE_VERSION_KEY, ALLOW_WRITE_VERSION);
      localStorage.setItem(ALLOW_WRITE_KEY, '1');
      return true;
    }
    const raw = localStorage.getItem(ALLOW_WRITE_KEY);
    if (raw === null) {
      return true;
    }
    return raw === '1' || raw === 'true';
  } catch {
    return true;
  }
}

export function writeStoredAllowWrite(enabled: boolean): void {
  try {
    localStorage.setItem(ALLOW_WRITE_VERSION_KEY, ALLOW_WRITE_VERSION);
    localStorage.setItem(ALLOW_WRITE_KEY, enabled ? '1' : '0');
  } catch {
    /* ignore */
  }
}

const LLM_SETTINGS_CACHE_KEY = 'llgraph-llm-settings-cache';

const SANDBOX_ENABLED_KEY = 'llgraph-sandbox-enabled';
const SANDBOX_ENABLED_VERSION_KEY = 'llgraph-sandbox-enabled-version';
const SANDBOX_ENABLED_VERSION = '1';

/** Web 工具栏「沙箱」偏好；缺省为 true。 */
export function readStoredSandboxEnabled(): boolean {
  try {
    const version = localStorage.getItem(SANDBOX_ENABLED_VERSION_KEY);
    if (version !== SANDBOX_ENABLED_VERSION) {
      localStorage.setItem(SANDBOX_ENABLED_VERSION_KEY, SANDBOX_ENABLED_VERSION);
      localStorage.setItem(SANDBOX_ENABLED_KEY, '1');
      return true;
    }
    const raw = localStorage.getItem(SANDBOX_ENABLED_KEY);
    if (raw === null) {
      return true;
    }
    return raw === '1' || raw === 'true';
  } catch {
    return true;
  }
}

export function writeStoredSandboxEnabled(enabled: boolean): void {
  try {
    localStorage.setItem(SANDBOX_ENABLED_VERSION_KEY, SANDBOX_ENABLED_VERSION);
    localStorage.setItem(SANDBOX_ENABLED_KEY, enabled ? '1' : '0');
  } catch {
    /* ignore */
  }
}

/** 刷新后立即展示上次模型列表（避免长时间「加载模型…」）。 */
export function readCachedLlmSettings(slug: string): LlmSettings | null {
  if (!slug) {
    return null;
  }
  try {
    const raw = localStorage.getItem(LLM_SETTINGS_CACHE_KEY);
    if (!raw) {
      return null;
    }
    const map = JSON.parse(raw) as Record<string, LlmSettings>;
    return map[slug] ?? null;
  } catch {
    return null;
  }
}

export function writeCachedLlmSettings(slug: string, settings: LlmSettings): void {
  if (!slug) {
    return;
  }
  try {
    const raw = localStorage.getItem(LLM_SETTINGS_CACHE_KEY);
    const map = raw ? (JSON.parse(raw) as Record<string, LlmSettings>) : {};
    map[slug] = settings;
    localStorage.setItem(LLM_SETTINGS_CACHE_KEY, JSON.stringify(map));
  } catch {
    /* ignore */
  }
}
