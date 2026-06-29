import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  type Capabilities,
  type LlmSettings,
  type SlashCatalogItem,
  type TreeNode,
  type Workspace,
} from '../api/client';
import { resolveWorkspaceSlug } from '../pages/console/storage';
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
        setAgents(t.agents ?? []);
        setPlans(t.plans ?? []);
        setTreeReadySlug(slug);
        setTreeLoading(false);
      })
      .catch(() => {
        if (seq !== treeFetchSeqRef.current) {
          return;
        }
        setTreeLoading(false);
        setTreeReadySlug(null);
      });
  }, [slug]);

  const refreshWorkspaceMeta = useCallback(() => {
    if (!slug) {
      return;
    }
    api.capabilities(slug, allowWrite).then(setCaps).catch(() => setCaps(null));
    api.llmSettings(slug).then(setLlmSettings).catch(() => setLlmSettings(null));
    api.slashCatalog(slug).then((r) => setSlashCatalog(r.items)).catch(() => setSlashCatalog([]));
  }, [slug, allowWrite, setCaps, setLlmSettings, setSlashCatalog]);

  const refreshTree = useCallback(() => {
    loadSessionTree();
    refreshWorkspaceMeta();
    refreshWorkspaces();
  }, [loadSessionTree, refreshWorkspaceMeta, refreshWorkspaces]);

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
