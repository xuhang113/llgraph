import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  type Capabilities,
  type LlmSettings,
  type SlashCatalogItem,
  type TreeNode,
  type Workspace,
} from '../api/client';
import {
  resolveWorkspaceSlug,
  readCachedLlmSettings,
  readStoredSandboxEnabled,
  writeCachedLlmSettings,
} from '../pages/console/storage';
import {
  mergeWorkspaceCatalog,
  readStoredRecentWorkspaces,
  readStoredWorkspaceMeta,
  readStoredWorkspaceSlug,
  writeStoredRecentWorkspaces,
  writeStoredWorkspaceMeta,
  workspaceLabelFromPath,
  type StoredWorkspaceMeta,
} from '../utils/workspaceStorage';

export type UseWorkspaceCatalogOptions = {
  allowWrite: boolean;
  setCaps: React.Dispatch<React.SetStateAction<Capabilities | null>>;
  setLlmSettings: React.Dispatch<React.SetStateAction<LlmSettings | null>>;
  setSlashCatalog: React.Dispatch<React.SetStateAction<SlashCatalogItem[]>>;
};

export function useWorkspaceCatalog({
  allowWrite,
  setCaps,
  setLlmSettings,
  setSlashCatalog,
}: UseWorkspaceCatalogOptions) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>(readStoredRecentWorkspaces);
  const [workspacesLoading, setWorkspacesLoading] = useState(true);
  const [slug, setSlug] = useState(readStoredWorkspaceSlug);
  const [agents, setAgents] = useState<TreeNode[]>([]);
  const [plans, setPlans] = useState<TreeNode[]>([]);
  const [treeLoading, setTreeLoading] = useState(false);
  const [treeReadySlug, setTreeReadySlug] = useState<string | null>(null);
  const treeFetchSeqRef = useRef(0);
  const workspacesFetchSeqRef = useRef(0);
  const warmedSlugRef = useRef<string | null>(null);

  const refreshWorkspaces = useCallback(() => {
    const seq = ++workspacesFetchSeqRef.current;
    setWorkspacesLoading(true);
    const attempt = (retriesLeft: number) => {
      api
        .workspaces()
        .then((d) => {
          if (seq !== workspacesFetchSeqRef.current) {
            return;
          }
          if (Array.isArray(d.workspaces)) {
            if (d.workspaces.length > 0) {
              setWorkspaces((prev) => {
                const pinned = readStoredWorkspaceSlug();
                const merged = mergeWorkspaceCatalog(d.workspaces, prev, pinned);
                writeStoredRecentWorkspaces(merged);
                return merged;
              });
            } else {
              setWorkspaces((prev) => (prev.length > 0 ? prev : d.workspaces));
            }
          }
          setWorkspacesLoading(false);
        })
        .catch(() => {
          if (seq !== workspacesFetchSeqRef.current) {
            return;
          }
          if (retriesLeft > 0) {
            window.setTimeout(() => attempt(retriesLeft - 1), 800);
          } else {
            setWorkspacesLoading(false);
          }
        });
    };
    attempt(5);
  }, []);

  const removeRecentWorkspace = useCallback((dismissSlug: string) => {
    setWorkspaces((prev) => {
      const next = prev.filter((w) => w.slug !== dismissSlug);
      writeStoredRecentWorkspaces(next);
      return next;
    });
  }, []);

  const loadSessionTree = useCallback(() => {
    if (!slug) {
      setTreeLoading(false);
      setTreeReadySlug(null);
      setAgents([]);
      setPlans([]);
      return;
    }
    const seq = ++treeFetchSeqRef.current;
    setTreeLoading(true);
    api
      .tree(slug)
      .then((t) => {
        if (seq !== treeFetchSeqRef.current) {
          return;
        }
        setAgents((prev) => {
          const loaded = t.agents ?? [];
          const loadedIds = new Set(loaded.map((n) => n.thread_id));
          const pending = prev.filter((n) => !loadedIds.has(n.thread_id));
          return [...pending, ...loaded];
        });
        setPlans(t.plans ?? []);
        setTreeReadySlug(slug);
        setTreeLoading(false);
        if (warmedSlugRef.current !== slug) {
          warmedSlugRef.current = slug;
          void api.warmRecentSessions(slug, allowWrite).catch(() => {});
        }
      })
      .catch(() => {
        if (seq !== treeFetchSeqRef.current) {
          return;
        }
        setTreeLoading(false);
        setTreeReadySlug(null);
      });
  }, [slug, allowWrite]);

  const refreshWorkspaceMeta = useCallback(() => {
    if (!slug) {
      return;
    }
    api.slashCatalog(slug).then((r) => setSlashCatalog(r.items)).catch(() => setSlashCatalog([]));
    window.setTimeout(() => {
      void api
        .capabilities(slug, allowWrite)
        .then(async (capsData) => {
          setCaps(capsData);
          if (
            readStoredSandboxEnabled() &&
            capsData.sandbox &&
            !capsData.sandbox.enabled &&
            capsData.sandbox.cli_override !== false
          ) {
            try {
              const res = await api.setSandbox(slug, true, '', allowWrite);
              if (res.sandbox) {
                setCaps((prev) => (prev ? { ...prev, sandbox: res.sandbox } : prev));
              }
            } catch {
              /* 沙箱后端不可用时保持未勾选 */
            }
          }
        })
        .catch(() => setCaps(null));
    }, 0);
  }, [slug, allowWrite, setCaps, setSlashCatalog]);

  const loadLlmSettings = useCallback(() => {
    if (!slug) {
      return;
    }
    const cached = readCachedLlmSettings(slug);
    if (cached) {
      setLlmSettings(cached);
    }
    const attempt = (retriesLeft: number) => {
      void api
        .llmSettings(slug)
        .then((settings) => {
          setLlmSettings(settings);
          writeCachedLlmSettings(slug, settings);
        })
        .catch(() => {
          if (retriesLeft > 0) {
            window.setTimeout(() => attempt(retriesLeft - 1), 600);
            return;
          }
          if (!cached) {
            setLlmSettings(null);
          }
        });
    };
    attempt(4);
  }, [slug, setLlmSettings]);

  const refreshTree = useCallback(() => {
    loadSessionTree();
    loadLlmSettings();
    refreshWorkspaceMeta();
    refreshWorkspaces();
  }, [loadSessionTree, loadLlmSettings, refreshWorkspaceMeta, refreshWorkspaces]);

  const refreshCaps = useCallback(() => {
    if (!slug) {
      return;
    }
    api.capabilities(slug, allowWrite).then(setCaps).catch(() => setCaps(null));
  }, [slug, allowWrite, setCaps]);

  useEffect(() => {
    refreshWorkspaces();
  }, [refreshWorkspaces]);

  /** 后端短暂不可达时，刷新后仍展示上次工作区名称/路径 */
  useEffect(() => {
    const onFocus = () => {
      if (workspaces.length === 0 && !workspacesLoading) {
        refreshWorkspaces();
      }
    };
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [workspaces.length, workspacesLoading, refreshWorkspaces]);

  const displayWorkspaces = useMemo((): Workspace[] => {
    const pinned = slug || readStoredWorkspaceSlug();
    const cached = readStoredRecentWorkspaces();
    return mergeWorkspaceCatalog(workspaces, cached, pinned);
  }, [workspaces, slug]);

  const slugKnownToCatalog = useCallback(
    (targetSlug: string) => {
      if (displayWorkspaces.some((w) => w.slug === targetSlug)) {
        return true;
      }
      const cached = readStoredWorkspaceMeta();
      return cached.slug === targetSlug && Boolean(cached.path);
    },
    [displayWorkspaces],
  );

  const workspaceDisplay = useMemo((): StoredWorkspaceMeta | null => {
    if (!slug) {
      return null;
    }
    const live = workspaces.find((w) => w.slug === slug);
    if (live) {
      return {
        slug: live.slug,
        path: live.path,
        label: workspaceLabelFromPath(live.path, live.slug),
      };
    }
    const cached = readStoredWorkspaceMeta();
    if (cached.slug === slug && (cached.path || cached.label)) {
      return cached;
    }
    return { slug, path: '', label: slug.slice(0, 8) };
  }, [slug, workspaces]);

  useEffect(() => {
    if (!workspaceDisplay?.slug || !workspaceDisplay.path) {
      return;
    }
    writeStoredWorkspaceMeta(workspaceDisplay);
  }, [workspaceDisplay]);

  useEffect(() => {
    if (workspaces.length === 0) {
      return;
    }
    setSlug((current) => {
      const next = resolveWorkspaceSlug(current, workspaces);
      if (next !== current) {
        treeFetchSeqRef.current += 1;
        setTreeReadySlug(null);
        setAgents([]);
        setPlans([]);
      }
      return next;
    });
  }, [workspaces]);

  /** workspaces 就绪后补拉 tree（避免刷新时 tree 先于 workspaces 失败且 slug 未变导致永不重试）。 */
  useEffect(() => {
    if (!slug) {
      return;
    }
    if (!slugKnownToCatalog(slug)) {
      return;
    }
    if (treeReadySlug === slug || treeLoading) {
      return;
    }
    loadSessionTree();
  }, [slug, slugKnownToCatalog, treeReadySlug, treeLoading, loadSessionTree]);

  useEffect(() => {
    if (!slug) {
      return;
    }
    api.touchWorkspace(slug).catch(() => {});
  }, [slug]);

  useEffect(() => {
    loadSessionTree();
  }, [loadSessionTree]);

  useEffect(() => {
    loadLlmSettings();
  }, [loadLlmSettings]);

  useEffect(() => {
    refreshWorkspaceMeta();
  }, [refreshWorkspaceMeta]);

  return {
    slug,
    setSlug,
    workspaces,
    displayWorkspaces,
    workspacesLoading,
    agents,
    setAgents,
    plans,
    setPlans,
    treeLoading,
    treeReadySlug,
    setTreeReadySlug,
    treeFetchSeqRef,
    workspaceDisplay,
    refreshWorkspaces,
    removeRecentWorkspace,
    loadSessionTree,
    refreshTree,
    refreshWorkspaceMeta,
    refreshCaps,
  };
}
