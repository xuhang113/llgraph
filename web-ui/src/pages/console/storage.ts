import type { Workspace } from '../../api/client';
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
