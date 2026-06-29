import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  type Capabilities,
  type LlmSettings,
  type PlanConfirmHistoryEntry,
  type PlanDetail,
  type SlashCatalogItem,
  type SurveySpec,
  type TreeNode,
} from '../../api/client';
import AgentToolbar from '../../components/console/AgentToolbar';
import ComposerDock from '../../components/console/ComposerDock';
import FileChangesPanel from '../../components/console/FileChangesPanel';
import CatalogPanel from '../../components/console/CatalogPanel';
import PlanMainPanel from '../../components/console/PlanMainPanel';
import WorkerMainPanel from '../../components/console/WorkerMainPanel';
import ChatComposer from '../../components/console/ChatComposer';
import { metaCommandModalTitle } from '../../utils/contextDisplay';
import ChatThread, { type ChatMessage } from '../../components/console/ChatThread';
import CursorRightPanel, { type RightPanelTab } from '../../components/console/CursorRightPanel';
import ConsoleOps, { type ConsoleOpsHandle } from '../../components/console/ConsoleOps';
import PanelResizeHandle from '../../components/console/PanelResizeHandle';
import CursorSidebar from '../../components/console/CursorSidebar';
import EditableSessionTitle from '../../components/console/EditableSessionTitle';
import CodeSearchPanel from '../../components/console/CodeSearchPanel';
import { useStickToBottomScroll } from '../../utils/useStickToBottomScroll';
import SurveyDialog, {
  PlanConfirmDialog,
  PlanConfirmReviewDialog,
  PlanConfirmSummaryChip,
  TaskStepConfirmDialog,
} from '../../components/console/SurveyDialogs';
import { planPlannerSubThread, planPlannerVersion } from '../../utils/planPlannerTrace';
import type { TraceStep, TraceTurn } from '../../types/trace';
import { buildDisplayTraceTurns, mergeTraceStepsUnique, stepsToPanelLogLines, traceStepsFingerprint } from '../../types/trace';
import { savePendingUserMessage } from '../../utils/pendingUserMessage';
import {
  clearComposerDraft,
  loadComposerDraft,
  saveComposerDraft,
} from '../../utils/composerDraft';
import {
  countPendingConfirms,
  dequeueConfirmHead,
  hasPendingKind,
  peekConfirmHead,
} from '../../utils/pendingConfirmQueue';
import { applyPendingConfirmHead, ingestPlanConfirmFromDetail } from '../../utils/pendingConfirmUi';
import {
  clearComposerImagesCache,
  getComposerImagesCache,
  setComposerImagesCache,
} from '../../utils/composerImagesCache';
import {
  getSessionChatCache,
  setSessionChatCache,
} from '../../utils/sessionChatCache';
import type { ComposerImage } from '../../types/chatImage';
import {
  loadTracePanelCache,
  saveTracePanelCache,
} from '../../utils/tracePanelStore';
import { useAppDialog } from '../../components/AppDialog';
import {
  writeStoredWorkspaceMeta,
  workspaceLabelFromPath,
} from '../../utils/workspaceStorage';
import {
  SIDEBAR_WIDTH_DEFAULT,
  SIDEBAR_WIDTH_MIN,
  SIDEBAR_WIDTH_MAX,
  RIGHT_PANEL_WIDTH_DEFAULT,
  RIGHT_PANEL_WIDTH_MIN,
  RIGHT_PANEL_WIDTH_MAX,
  SSE_STALE_MS,
  SSE_TRACE_POLL_SKIP_MS,
} from './constants';
import {
  readStoredPanelWidth,
  clampPanelWidth,
  readStoredSessionThread,
  writeStoredSessionThread,
} from './storage';
import { planExecutionAllowWrite } from './planHelpers';
import {
  parseTraceStep,
  panelLinesFromTexts,
  releaseTurnOpen,
  loadTraceTurnsFromRemote,
  mergeLiveTraceIntoPanel,
  isRemoteTraceAhead,
  maxStepId,
  preferRicherTraceCache,
} from './traceUtils';
import { parseWorkerMessages } from './workerUtils';
import { shouldSuppressSessionTrace } from './sseHelpers';
import { SSE_TRACE_CONTENT_TYPES } from './constants';
import { bumpSidebarSession } from './sidebarUtils';
import type { TraceLine } from './types';
import { useWorkspaceCatalog } from '../../hooks/useWorkspaceCatalog';
import { createHandleSSEEvent } from '../../hooks/useConsoleSSE';
import { useSessionHistory } from '../../hooks/useSessionHistory';
import { usePlanSession } from '../../hooks/usePlanSession';



export default function ConsolePage() {
  const { alert, confirm, prompt } = useAppDialog();
  const [selected, setSelected] = useState<TreeNode | null>(null);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [llmSettings, setLlmSettings] = useState<LlmSettings | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const messagesRef = useRef<ChatMessage[]>([]);
  const [traceLines, setTraceLines] = useState<TraceLine[]>([]);
  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);
  /** 右侧 Trace 面板：最后一轮完整日志（独立于聊天区） */
  const [panelTraceLines, setPanelTraceLines] = useState<TraceLine[]>([]);
  const [panelTraceSteps, setPanelTraceSteps] = useState<TraceStep[]>([]);
  const [panelTraceTurns, setPanelTraceTurns] = useState<TraceTurn[]>([]);
  const [slashCatalog, setSlashCatalog] = useState<SlashCatalogItem[]>([]);
  const [input, setInput] = useState('');
  const [composerImages, setComposerImages] = useState<ComposerImage[]>([]);
  const [busy, setBusy] = useState(false);
  const [allowWrite, setAllowWrite] = useState(false);

  const {
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
    refreshTree,
    refreshCaps,
  } = useWorkspaceCatalog({ allowWrite, setCaps, setLlmSettings, setSlashCatalog });

  const [planDetail, setPlanDetail] = useState<PlanDetail | null>(null);
  const [survey, setSurvey] = useState<SurveySpec | null>(null);
  const [planConfirm, setPlanConfirm] = useState<Record<string, unknown> | null>(null);
  const [planConfirmReview, setPlanConfirmReview] = useState<PlanConfirmHistoryEntry | null>(null);
  const [taskStepConfirm, setTaskStepConfirm] = useState<string | null>(null);
  const [fileChangesTick, setFileChangesTick] = useState(0);
  const [contextRefreshSignal, setContextRefreshSignal] = useState(0);
  const bumpContextRefresh = useCallback(() => {
    setContextRefreshSignal((n) => n + 1);
  }, []);
  const [streamText, setStreamText] = useState('');
  const [thinkingText, setThinkingText] = useState('');
  const [traceActivitySec, setTraceActivitySec] = useState(0);
  const traceTurnStartRef = useRef(0);
  const traceLiveTsRef = useRef<Map<string, string>>(new Map());
  const lastSessionEventAtRef = useRef<Map<string, number>>(new Map());
  const [catalogOpen, setCatalogOpen] = useState<'skills' | 'rules' | 'tools' | null>(
    null,
  );
  const [rightPanelOpen, setRightPanelOpen] = useState(() => {
    try {
      return localStorage.getItem('llgraph-right-panel') !== '0';
    } catch {
      return true;
    }
  });
  const [rightPanelRequestedTab, setRightPanelRequestedTab] = useState<RightPanelTab | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(() =>
    readStoredPanelWidth(
      'llgraph-sidebar-width',
      SIDEBAR_WIDTH_DEFAULT,
      SIDEBAR_WIDTH_MIN,
      SIDEBAR_WIDTH_MAX,
    ),
  );
  const [rightPanelWidth, setRightPanelWidth] = useState(() =>
    readStoredPanelWidth(
      'llgraph-right-panel-width',
      RIGHT_PANEL_WIDTH_DEFAULT,
      RIGHT_PANEL_WIDTH_MIN,
      RIGHT_PANEL_WIDTH_MAX,
    ),
  );
  const [panelResizing, setPanelResizing] = useState<'sidebar' | 'right' | null>(null);
  const [codeSearchOpen, setCodeSearchOpen] = useState(false);
  const [multiSelectMode, setMultiSelectMode] = useState(false);
  const [selectedSessionIds, setSelectedSessionIds] = useState<Set<string>>(() => new Set());
  const streamedRef = useRef(false);
  const traceFlushedRef = useRef(false);
  const traceStepsRef = useRef<TraceStep[]>([]);
  const traceLinesRef = useRef<TraceLine[]>([]);
  const panelTraceLinesRef = useRef<TraceLine[]>([]);
  const panelTraceStepsRef = useRef<TraceStep[]>([]);
  const panelTraceTurnsRef = useRef<TraceTurn[]>([]);
  const onSSEHandlerRef = useRef<(event: Record<string, unknown>) => void>(() => {});
  const abortRef = useRef<AbortController | null>(null);
  const streamAbortRef = useRef<Map<string, AbortController>>(new Map());
  const streamLastEventAtRef = useRef<Map<string, number>>(new Map());
  const selectedRef = useRef<TreeNode | null>(null);
  const runningThreadsRef = useRef<Set<string>>(new Set());
  const turnOpenRef = useRef<Set<string>>(new Set());
  const stoppedThreadsRef = useRef<Set<string>>(new Set());
  const planSubsRef = useRef<Map<string, () => void>>(new Map());
  const sessionSubsRef = useRef<Map<string, () => void>>(new Map());
  const handleSSEEventRef = useRef<(event: Record<string, unknown>, eventThread: string) => void>(
    () => {},
  );
  const chatEndRef = useRef<HTMLDivElement>(null);
  const consoleOpsRef = useRef<ConsoleOpsHandle>(null);
  const sidebarWidthRef = useRef(sidebarWidth);
  const rightPanelWidthRef = useRef(rightPanelWidth);
  sidebarWidthRef.current = sidebarWidth;
  rightPanelWidthRef.current = rightPanelWidth;
  const pollRef = useRef<number | null>(null);
  /** 本轮 awaiting_confirm 已自动展示过确认框，避免 loadHistory + effect 重复弹 */
  const confirmAutoShownRef = useRef<string | null>(null);
  const maybePromptPlanConfirmRef = useRef<(detail: PlanDetail, threadId: string) => void>(() => {});
  const sessionRestoreSlugRef = useRef<string | null>(null);
  const plannerVerRef = useRef<Map<string, number>>(new Map());
  const surveyDismissedRef = useRef<string | null>(null);
  const taskStepDismissedRef = useRef<string | null>(null);
  const planStopInFlightRef = useRef(false);

  const fileChangesConfig = useMemo(() => {
    if (!slug || !selected) {
      return null;
    }
    if (selected.kind === 'agent') {
      return {
        mode: 'agent' as const,
        sessionThreadId: selected.thread_id,
      };
    }
    if (selected.kind === 'plan') {
      return {
        mode: 'plan' as const,
        sessionThreadId: selected.thread_id,
        planThreadId: selected.thread_id,
      };
    }
    if (selected.kind === 'worker') {
      const planThread = selected.thread_id.split(':worker:')[0];
      const taskId = selected.task_id || selected.thread_id.split(':worker:')[1] || '';
      return {
        mode: 'worker' as const,
        sessionThreadId: selected.thread_id,
        planThreadId: planThread,
        taskId,
      };
    }
    return null;
  }, [slug, selected]);

  useEffect(() => {
    document.documentElement.classList.add('cursor-theme');
    return () => document.documentElement.classList.remove('cursor-theme');
  }, []);

  selectedRef.current = selected;

  const planDetailRef = useRef<PlanDetail | null>(null);
  planDetailRef.current = planDetail;

  const applyTraceToViewIfSelected = useCallback(
    (
      threadId: string,
      panelLines: TraceLine[],
      panelSteps: TraceStep[],
      panelTurns: TraceTurn[] = [],
    ) => {
      const sel = selectedRef.current;
      if (!sel) {
        return;
      }
      const traceThread =
        sel.kind === 'worker'
          ? sel.thread_id
          : sel.kind === 'agent'
            ? sel.thread_id
            : null;
      if (traceThread === threadId) {
        setPanelTraceLines(panelLines);
        setPanelTraceSteps(panelSteps);
        setPanelTraceTurns(panelTurns);
        panelTraceLinesRef.current = panelLines;
        panelTraceStepsRef.current = panelSteps;
        panelTraceTurnsRef.current = panelTurns;
        return;
      }
      if (sel.kind === 'plan' && threadId.includes(':planner:')) {
        setPanelTraceLines(panelLines);
        setPanelTraceSteps(panelSteps);
        setPanelTraceTurns(panelTurns);
        panelTraceLinesRef.current = panelLines;
        panelTraceStepsRef.current = panelSteps;
        panelTraceTurnsRef.current = panelTurns;
      }
    },
    [],
  );

  const syncRemoteTrace = useCallback(
    async (threadId: string) => {
      if (!slug || !threadId) {
        return null;
      }
      try {
        const remote = await api.lastTrace(slug, threadId);
        const loaded = loadTraceTurnsFromRemote(remote);
        let panelSteps = loaded.currentSteps;
        let panelTurns = loaded.completed;
        let panelLines: TraceLine[] = remote.log_lines?.length
          ? panelLinesFromTexts(remote.log_lines)
          : stepsToPanelLogLines(panelSteps);
        saveTracePanelCache(
          slug,
          threadId,
          panelLines.map((l) => l.text),
          panelSteps,
        );
        const curPanelSteps = panelTraceStepsRef.current;
        const curPanelLines = panelTraceLinesRef.current;
        const panelEmpty = curPanelSteps.length === 0 && curPanelLines.length === 0;
        const remoteRicher = isRemoteTraceAhead(
          panelSteps,
          panelLines.length,
          curPanelSteps,
          curPanelLines.length,
        );
        const agentRunning =
          runningThreadsRef.current.has(threadId) ||
          (selectedRef.current?.kind === 'agent' &&
            selectedRef.current.thread_id === threadId);
        const remoteMax = maxStepId(panelSteps);
        const currentMax = maxStepId(curPanelSteps);
        const remoteFreshWhileRunning =
          agentRunning && remoteMax >= currentMax && panelSteps.length > 0;
        if (panelEmpty || remoteRicher || remoteFreshWhileRunning) {
          const liveTs = String(remote.live_ts ?? '');
          if (liveTs) {
            traceLiveTsRef.current.set(threadId, liveTs);
          }
          applyTraceToViewIfSelected(threadId, panelLines, panelSteps, panelTurns);
        }
        return { panelLines, panelSteps, panelTurns };
      } catch {
        return null;
      }
    },
    [slug, applyTraceToViewIfSelected],
  );

  const flushPlannerTraceCacheToPanel = useCallback(
    (sub: string) => {
      if (!slug || !sub) {
        return;
      }
      const cached = loadTracePanelCache(slug, sub);
      if (!cached || (cached.log_lines.length === 0 && cached.steps.length === 0)) {
        return;
      }
      applyTraceToViewIfSelected(
        sub,
        panelLinesFromTexts(cached.log_lines),
        cached.steps,
      );
    },
    [slug, applyTraceToViewIfSelected],
  );

  const ensureSessionSubscription = useCallback(
    (sessionThread: string) => {
      if (!slug || !sessionThread || sessionSubsRef.current.has(sessionThread)) {
        return;
      }
      const traceTypes = new Set([
        'trace_line',
        'trace_step',
        'trace_activity',
        'turn_start',
        'thinking_delta',
        'stream_delta',
        'stream_end',
        'turn_done',
        'survey',
        'end',
      ]);
      const unsub = api.subscribeSessionEvents(slug, sessionThread, (ev) => {
        const t = String(ev.type || '');
        const tid = String(ev.thread_id || sessionThread);
        if (t !== 'ping' && SSE_TRACE_CONTENT_TYPES.has(t)) {
          lastSessionEventAtRef.current.set(tid, Date.now());
        }
        if (t === 'ping') {
          return;
        }
        if (
          traceTypes.has(t) &&
          shouldSuppressSessionTrace(
            tid,
            streamAbortRef.current,
            streamLastEventAtRef.current,
            selectedRef.current,
          )
        ) {
          return;
        }
        if (t === 'turn_start') {
          runningThreadsRef.current.add(tid);
          lastSessionEventAtRef.current.set(tid, Date.now());
        } else if (t === 'end') {
          runningThreadsRef.current.delete(tid);
          void syncRemoteTrace(tid);
        }
        if (traceTypes.has(t)) {
          onSSEHandlerRef.current({ ...ev, thread_id: tid });
        }
      });
      sessionSubsRef.current.set(sessionThread, unsub);
    },
    [slug, syncRemoteTrace, applyTraceToViewIfSelected, flushPlannerTraceCacheToPanel],
  );

  const ensurePlanPlannerSubscription = useCallback(
    (planThread: string, versionOverride?: number) => {
      const detail = planDetailRef.current;
      const baseVer = planPlannerVersion(detail);
      const ver =
        typeof versionOverride === 'number' && versionOverride > 0
          ? versionOverride
          : baseVer;
      const prevVer = plannerVerRef.current.get(planThread);
      if (prevVer != null && prevVer !== ver) {
        const oldThread = planPlannerSubThread(planThread, prevVer);
        sessionSubsRef.current.get(oldThread)?.();
        sessionSubsRef.current.delete(oldThread);
      }
      plannerVerRef.current.set(planThread, ver);
      const plannerThread = planPlannerSubThread(planThread, ver);
      ensureSessionSubscription(plannerThread);
      return plannerThread;
    },
    [ensureSessionSubscription],
  );

  const ensurePlanSubscription = useCallback(
    (planThread: string) => {
      if (!slug || planSubsRef.current.has(planThread)) {
        return;
      }
      const forwardTypes = new Set([
        'end',
        'error',
        'interrupt',
        'plan_job',
        'plan_state',
        'plan_done',
      ]);
      const unsub = api.subscribePlanEvents(slug, planThread, (ev) => {
        const t = String(ev.type || '');
        // POST SSE 与长连接订阅同一 channel；POST 仍活跃时由 POST 独占 trace，避免重复渲染
        if (
          forwardTypes.has(t) &&
          shouldSuppressSessionTrace(
            planThread,
            streamAbortRef.current,
            streamLastEventAtRef.current,
            selectedRef.current,
          )
        ) {
          return;
        }
        if (forwardTypes.has(t)) {
          onSSEHandlerRef.current({ ...ev, thread_id: ev.thread_id || planThread });
        }
        if (t === 'plan_state' || t === 'turn_done' || t === 'end' || t === 'subscribed') {
          if (
            selectedRef.current?.thread_id === planThread &&
            selectedRef.current?.kind === 'plan'
          ) {
            api.plan(slug, planThread).then((detail) => {
              setPlanDetail(detail);
              if (planExecutionAllowWrite(detail)) {
                setAllowWrite(true);
              }
            }).catch(() => {});
            refreshTree();
          }
        }
      });
      planSubsRef.current.set(planThread, unsub);
    },
    [slug, refreshTree],
  );

  const maybeReleasePlanSubscription = useCallback((planThread: string) => {
    if (selectedRef.current?.thread_id === planThread) {
      return;
    }
    if (runningThreadsRef.current.has(planThread)) {
      return;
    }
    planSubsRef.current.get(planThread)?.();
    planSubsRef.current.delete(planThread);
  }, []);

  const ingestSSE = useCallback((event: Record<string, unknown>, defaultThread?: string) => {
    const threadId = String(event.thread_id || defaultThread || '');
    if (!threadId) {
      return;
    }
    handleSSEEventRef.current(event, threadId);
  }, []);

  const bindSSE = useCallback(
    (threadId: string) => (ev: Record<string, unknown>) => {
      streamLastEventAtRef.current.set(threadId, Date.now());
      ingestSSE(ev, threadId);
    },
    [ingestSSE],
  );

  const resubscribeSession = useCallback(
    (sessionThread: string) => {
      if (!slug || !sessionThread) {
        return;
      }
      sessionSubsRef.current.get(sessionThread)?.();
      sessionSubsRef.current.delete(sessionThread);
      ensureSessionSubscription(sessionThread);
      void syncRemoteTrace(sessionThread);
    },
    [slug, ensureSessionSubscription, syncRemoteTrace],
  );

  const ensureRunningSessionSubscriptions = useCallback(() => {
    const now = Date.now();
    for (const tid of runningThreadsRef.current) {
      const last = lastSessionEventAtRef.current.get(tid) ?? 0;
      if (last > 0 && now - last > SSE_STALE_MS) {
        resubscribeSession(tid);
      } else {
        ensureSessionSubscription(tid);
      }
      if (tid.startsWith('plan-')) {
        ensurePlanSubscription(tid);
        ensurePlanPlannerSubscription(tid);
      }
    }
  }, [
    ensureSessionSubscription,
    ensurePlanSubscription,
    ensurePlanPlannerSubscription,
    resubscribeSession,
  ]);

  const syncAgentRunningState = useCallback(
    async (threadId: string, opts: { updateBusy?: boolean } = {}) => {
      if (!slug || !threadId) {
        return false;
      }
      const updateBusy = opts.updateBusy !== false;
      try {
        const meta = await api.sessionMeta(slug, threadId);
        const running = Boolean(meta.running ?? meta.lock?.owner === 'web');
        const localTurn = runningThreadsRef.current.has(threadId);
        if (running) {
          runningThreadsRef.current.add(threadId);
          if (updateBusy && selectedRef.current?.thread_id === threadId) {
            setBusy(true);
          }
        } else if (!localTurn) {
          // 本地 turn_start 已标记执行中时不因 meta 短暂为 false 清 busy（长工具阻塞常见）
          runningThreadsRef.current.delete(threadId);
          streamAbortRef.current.delete(threadId);
          streamLastEventAtRef.current.delete(threadId);
          if (updateBusy && selectedRef.current?.thread_id === threadId) {
            setBusy(false);
          }
        }
        return running || localTurn;
      } catch {
        return runningThreadsRef.current.has(threadId);
      }
    },
    [slug],
  );

  const resumeSessionAfterSelect = useCallback(
    async (node: TreeNode) => {
      if (!slug || selectedRef.current?.thread_id !== node.thread_id) {
        return;
      }
      resubscribeSession(node.thread_id);
      ensureRunningSessionSubscriptions();

      if (node.kind === 'agent') {
        await syncAgentRunningState(node.thread_id);
      } else if (node.kind === 'plan') {
        try {
          const detail = await api.plan(slug, node.thread_id);
          if (selectedRef.current?.thread_id !== node.thread_id) {
            return;
          }
          if (detail.job?.running || (detail as { lock?: { owner?: string } }).lock?.owner === 'web') {
            runningThreadsRef.current.add(node.thread_id);
            setBusy(true);
            ensurePlanSubscription(node.thread_id);
          }
        } catch {
          /* ignore */
        }
      } else if (node.kind === 'worker') {
        resubscribeSession(node.thread_id);
      }

      const traceThread =
        node.kind === 'worker' ? node.thread_id : node.thread_id;
      const loaded = await syncRemoteTrace(traceThread);
      if (!loaded || selectedRef.current?.thread_id !== node.thread_id) {
        return;
      }
      const picked = preferRicherTraceCache(
        slug,
        traceThread,
        loaded.panelLines,
        loaded.panelSteps,
      );
      if (picked.lines.length || picked.steps.length) {
        setPanelTraceLines(picked.lines);
        setPanelTraceSteps(picked.steps);
        panelTraceLinesRef.current = picked.lines;
        panelTraceStepsRef.current = picked.steps;
      } else if (
        panelTraceLinesRef.current.length === 0 &&
        panelTraceStepsRef.current.length === 0
      ) {
        const cached = loadTracePanelCache(slug, traceThread);
        if (cached && (cached.log_lines.length > 0 || cached.steps.length > 0)) {
          const cacheLines = panelLinesFromTexts(cached.log_lines);
          setPanelTraceLines(cacheLines);
          setPanelTraceSteps(cached.steps);
          panelTraceLinesRef.current = cacheLines;
          panelTraceStepsRef.current = cached.steps;
        }
      }
    },
    [
      slug,
      resubscribeSession,
      ensureRunningSessionSubscriptions,
      syncRemoteTrace,
      syncAgentRunningState,
    ],
  );

  const { loadHistory, historyLoading } = useSessionHistory({
    slug,
    maybePromptPlanConfirm: (detail, threadId) => maybePromptPlanConfirmRef.current(detail, threadId),
    ensurePlanSubscription,
    ensurePlanPlannerSubscription,
    ensureSessionSubscription,
    resumeSessionAfterSelect,
    syncAgentRunningState,
    runningThreadsRef,
    turnOpenRef,
    traceLinesRef,
    traceStepsRef,
    panelTraceLinesRef,
    panelTraceStepsRef,
    panelTraceTurnsRef,
    surveyDismissedRef,
    taskStepDismissedRef,
    streamAbortRef,
    setAllowWrite,
    setCaps,
    setPlanDetail,
    setSurvey,
    setPlanConfirm,
    setTaskStepConfirm,
    setMessages,
    setTraceLines,
    setTraceSteps,
    setStreamText,
    setThinkingText,
    setPanelTraceLines,
    setPanelTraceSteps,
    setPanelTraceTurns,
    setBusy,
    bumpContextRefresh,
  });

  const reconnectSessionSubscriptions = useCallback(() => {
    const threads = new Set<string>();
    for (const tid of runningThreadsRef.current) {
      threads.add(tid);
    }
    const sel = selectedRef.current;
    if (sel) {
      threads.add(sel.thread_id);
      if (sel.kind === 'worker') {
        const main = sel.thread_id.split(':worker:')[0];
        if (main) {
          threads.add(main);
        }
      }
    }
    for (const tid of threads) {
      if (tid) {
        resubscribeSession(tid);
      }
    }
    for (const [planThread, unsub] of planSubsRef.current) {
      if (runningThreadsRef.current.has(planThread)) {
        unsub();
        planSubsRef.current.delete(planThread);
        ensurePlanSubscription(planThread);
      }
    }
  }, [resubscribeSession, ensurePlanSubscription]);

  const beginStream = useCallback(
    (threadId: string) => {
      runningThreadsRef.current.add(threadId);
      ensureSessionSubscription(threadId);
      if (threadId.startsWith('plan-')) {
        ensurePlanPlannerSubscription(threadId);
        ensurePlanSubscription(threadId);
      }
      const ac = new AbortController();
      streamAbortRef.current.set(threadId, ac);
      abortRef.current = ac;
      return ac;
    },
    [ensureSessionSubscription, ensurePlanPlannerSubscription, ensurePlanSubscription],
  );

  useEffect(() => {
    if (!slug) {
      return;
    }
    const id = window.setInterval(() => {
      ensureRunningSessionSubscriptions();
      const sel = selectedRef.current;
      if (sel?.kind === 'agent' && slug) {
        const last = lastSessionEventAtRef.current.get(sel.thread_id) ?? 0;
        if (last <= 0 || Date.now() - last >= SSE_TRACE_POLL_SKIP_MS) {
          void syncRemoteTrace(sel.thread_id);
        }
      }
      for (const tid of runningThreadsRef.current) {
        if (sel?.kind === 'agent' && sel.thread_id === tid) {
          continue;
        }
        const last = lastSessionEventAtRef.current.get(tid) ?? 0;
        if (last > 0 && Date.now() - last < SSE_TRACE_POLL_SKIP_MS) {
          continue;
        }
        void syncRemoteTrace(tid);
        if (tid.startsWith('plan-')) {
          const ver = planPlannerVersion(planDetailRef.current);
          void syncRemoteTrace(planPlannerSubThread(tid, ver));
        }
      }
    }, 2000);
    return () => window.clearInterval(id);
  }, [slug, syncRemoteTrace, ensureRunningSessionSubscriptions]);

  useEffect(() => {
    if (!slug || selected?.kind !== 'agent') {
      return;
    }
    const threadId = selected.thread_id;
    void syncAgentRunningState(threadId);
    const id = window.setInterval(() => {
      void syncAgentRunningState(threadId);
    }, 3000);
    return () => window.clearInterval(id);
  }, [slug, selected?.kind, selected?.thread_id, syncAgentRunningState]);

  useEffect(() => {
    if (!slug) {
      return;
    }
    const onVisibility = () => {
      if (document.visibilityState !== 'visible') {
        return;
      }
      for (const tid of runningThreadsRef.current) {
        void syncRemoteTrace(tid);
      }
      reconnectSessionSubscriptions();
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [slug, syncRemoteTrace, reconnectSessionSubscriptions]);

  useEffect(() => {
    if (!busy) {
      return;
    }
    if (!traceTurnStartRef.current) {
      traceTurnStartRef.current = Date.now();
    }
    const id = window.setInterval(() => {
      if (!traceTurnStartRef.current) {
        return;
      }
      setTraceActivitySec(Math.floor((Date.now() - traceTurnStartRef.current) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [busy, selected?.thread_id]);

  useEffect(() => {
    if (!slug || !planDetail) {
      return;
    }
    const planThread =
      selected?.kind === 'plan'
        ? selected.thread_id
        : selected?.kind === 'worker'
          ? selected.thread_id.split(':worker:')[0]
          : planDetail.thread_id;
    if (!planThread) {
      return;
    }
    for (const task of planDetail.tasks || []) {
      if (String(task.status) === 'running') {
        ensureSessionSubscription(`${planThread}:worker:${task.id}`);
      }
    }
  }, [slug, planDetail, selected, ensureSessionSubscription]);

  useEffect(() => {
    if (!slug || selected?.kind !== 'plan' || !planDetail) {
      return;
    }
    const planThread = selected.thread_id;
    const plannerThread = ensurePlanPlannerSubscription(planThread);
    void (async () => {
      const loaded = await syncRemoteTrace(plannerThread);
      if (!loaded) {
        return;
      }
      const picked = preferRicherTraceCache(
        slug,
        plannerThread,
        loaded.panelLines,
        loaded.panelSteps,
      );
      if (picked.lines.length || picked.steps.length) {
        applyTraceToViewIfSelected(plannerThread, picked.lines, picked.steps);
      }
    })();
  }, [
    slug,
    selected?.kind,
    selected?.thread_id,
    planDetail?.phase,
    planDetail?.job?.running,
    planDetail?.plan_state?.plan_version,
    planDetail?.plan,
    ensurePlanPlannerSubscription,
    syncRemoteTrace,
    applyTraceToViewIfSelected,
  ]);

  useEffect(() => {
    if (!slug || selected?.kind !== 'plan') {
      return;
    }
    const planThread = selected.thread_id;
    const syncBusy = () => {
      api
        .plan(slug, planThread)
        .then((detail) => {
          setPlanDetail(detail);
          if (detail.job?.running) {
            runningThreadsRef.current.add(planThread);
            setBusy(true);
          } else {
            streamAbortRef.current.delete(planThread);
            runningThreadsRef.current.delete(planThread);
            setBusy(false);
          }
        })
        .catch(() => {});
    };
    syncBusy();
    const id = window.setInterval(syncBusy, 3000);
    return () => window.clearInterval(id);
  }, [slug, selected?.kind, selected?.thread_id]);

  useEffect(() => {
    if (!slug || (selected?.kind !== 'plan' && selected?.kind !== 'worker')) {
      return;
    }
    const planThread =
      selected.kind === 'worker'
        ? selected.thread_id.split(':worker:')[0]
        : selected.thread_id;
    ensurePlanSubscription(planThread);
  }, [slug, selected?.kind, selected?.thread_id, ensurePlanSubscription]);

  useEffect(() => {
    if (!slug) {
      return;
    }
    for (const unsub of planSubsRef.current.values()) {
      unsub();
    }
    planSubsRef.current.clear();
    for (const unsub of sessionSubsRef.current.values()) {
      unsub();
    }
    sessionSubsRef.current.clear();
    plannerVerRef.current.clear();
  }, [slug]);

  useEffect(
    () => () => {
      for (const unsub of planSubsRef.current.values()) {
        unsub();
      }
      planSubsRef.current.clear();
      for (const unsub of sessionSubsRef.current.values()) {
        unsub();
      }
      sessionSubsRef.current.clear();
    },
    [],
  );

  useEffect(() => {
    traceStepsRef.current = traceSteps;
  }, [traceSteps]);

  useEffect(() => {
    traceLinesRef.current = traceLines;
  }, [traceLines]);

  useEffect(() => {
    panelTraceLinesRef.current = panelTraceLines;
  }, [panelTraceLines]);

  useEffect(() => {
    panelTraceStepsRef.current = panelTraceSteps;
  }, [panelTraceSteps]);

  const mainScrollPinDeps = useMemo(
    () => [
      messages.length,
      streamText.length,
      traceStepsFingerprint(panelTraceSteps),
      traceStepsFingerprint(traceSteps),
      panelTraceLines.map((l) => l.text).join('\n').length,
      traceLines.map((l) => l.text).join('\n').length,
      thinkingText.length,
      busy,
    ],
    [
      messages.length,
      streamText.length,
      panelTraceSteps,
      traceSteps,
      panelTraceLines,
      traceLines,
      thinkingText,
      busy,
    ],
  );
  const mainScroll = useStickToBottomScroll<HTMLDivElement>(mainScrollPinDeps, {
    enabled: true,
    resetKey: selected?.thread_id ?? '',
  });

  useEffect(() => {
    try {
      localStorage.setItem('llgraph-right-panel', rightPanelOpen ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [rightPanelOpen]);

  useEffect(() => {
    if (!panelResizing) {
      return;
    }
    const onMove = (e: MouseEvent) => {
      if (panelResizing === 'sidebar') {
        setSidebarWidth(clampPanelWidth(e.clientX, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX));
        return;
      }
      setRightPanelWidth(
        clampPanelWidth(
          window.innerWidth - e.clientX,
          RIGHT_PANEL_WIDTH_MIN,
          RIGHT_PANEL_WIDTH_MAX,
        ),
      );
    };
    const onUp = () => {
      setPanelResizing(null);
      try {
        localStorage.setItem('llgraph-sidebar-width', String(sidebarWidthRef.current));
        localStorage.setItem('llgraph-right-panel-width', String(rightPanelWidthRef.current));
      } catch {
        /* ignore */
      }
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    document.body.classList.add('cursor-panel-resizing');
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.classList.remove('cursor-panel-resizing');
    };
  }, [panelResizing]);

  useEffect(() => {
    if (!slug || treeReadySlug !== slug || agents.length === 0) {
      return;
    }
    if (sessionRestoreSlugRef.current === slug) {
      return;
    }
    sessionRestoreSlugRef.current = slug;
    const lastThread = readStoredSessionThread(slug);
    if (!lastThread) {
      return;
    }
    const node =
      agents.find((a) => a.thread_id === lastThread) ??
      plans.find((p) => p.thread_id === lastThread);
    if (!node) {
      return;
    }
    selectedRef.current = node;
    setSelected(node);
    void loadHistory(node);
  }, [slug, treeReadySlug, agents, plans, loadHistory]);

  useEffect(() => {
    const current = selectedRef.current;
    if (!current || current.kind === 'worker') {
      return;
    }
    const updated =
      agents.find((a) => a.thread_id === current.thread_id) ??
      plans.find((p) => p.thread_id === current.thread_id);
    if (!updated) {
      return;
    }
    if (
      updated.title !== current.title ||
      (updated.title_full ?? '') !== (current.title_full ?? '')
    ) {
      const merged = { ...current, title: updated.title, title_full: updated.title_full };
      selectedRef.current = merged;
      setSelected(merged);
    }
  }, [agents, plans]);

  const handleSelect = (node: TreeNode) => {
    const prev = selectedRef.current;
    if (prev && slug && prev.thread_id !== node.thread_id) {
      saveComposerDraft(slug, prev.thread_id, input);
      setComposerImagesCache(slug, prev.thread_id, composerImages);
      setSessionChatCache(slug, prev.thread_id, messagesRef.current);
      persistPanelTraceSnapshot(prev.thread_id);
      streamLastEventAtRef.current.delete(prev.thread_id);
      const prevMain =
        prev.kind === 'worker' ? prev.thread_id.split(':worker:')[0] : prev.thread_id;
      if (prevMain) {
        streamLastEventAtRef.current.delete(prevMain);
      }
    }
    setCatalogOpen(null);
    selectedRef.current = node;
    setSelected(node);
    setStreamText('');
    setThinkingText('');
    setTraceLines([]);
    setTraceSteps([]);
    traceLinesRef.current = [];
    traceStepsRef.current = [];
    setInput(slug ? loadComposerDraft(slug, node.thread_id) : '');
    setComposerImages(slug ? getComposerImagesCache(slug, node.thread_id) : []);
    const cachedChat = slug ? getSessionChatCache(slug, node.thread_id) : undefined;
    setMessages(cachedChat ?? []);
    if (slug) {
      const panelCached = loadTracePanelCache(slug, node.thread_id);
      if (panelCached) {
        const lines = panelLinesFromTexts(panelCached.log_lines);
        setPanelTraceLines(lines);
        setPanelTraceSteps(panelCached.steps);
        setPanelTraceTurns([]);
        panelTraceLinesRef.current = lines;
        panelTraceStepsRef.current = panelCached.steps;
        panelTraceTurnsRef.current = [];
      } else {
        setPanelTraceLines([]);
        setPanelTraceSteps([]);
        setPanelTraceTurns([]);
        panelTraceLinesRef.current = [];
        panelTraceStepsRef.current = [];
        panelTraceTurnsRef.current = [];
      }
    }
    setBusy(runningThreadsRef.current.has(node.thread_id));
    if (slug) {
      writeStoredSessionThread(slug, node.thread_id);
    }
    loadHistory(node);
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (node.kind === 'worker' && slug) {
      const planThread = node.thread_id.split(':worker:')[0];
      const taskId = node.task_id || '';
      pollRef.current = window.setInterval(async () => {
        try {
          const [data, detail] = await Promise.all([
            api.worker(slug, planThread, taskId),
            api.plan(slug, planThread),
          ]);
          setPlanDetail(detail);
          const { chat, traces, traceSteps } = parseWorkerMessages(data);
          setMessages(chat);
          if (selectedRef.current?.thread_id === node.thread_id) {
            try {
              const remote = await api.lastTrace(slug, node.thread_id);
              let pollLines: TraceLine[] = [];
              let pollSteps: TraceStep[] = [];
              if (remote.steps?.length) {
                pollSteps = remote.steps.map((s) => parseTraceStep(s));
              }
              if (remote.log_lines?.length) {
                pollLines = panelLinesFromTexts(remote.log_lines);
              } else if (pollSteps.length > 0) {
                pollLines = stepsToPanelLogLines(pollSteps);
              }
              const picked = preferRicherTraceCache(slug, node.thread_id, pollLines, pollSteps);
              if (picked.lines.length || picked.steps.length) {
                setPanelTraceLines(picked.lines);
                setPanelTraceSteps(picked.steps);
                panelTraceLinesRef.current = picked.lines;
                panelTraceStepsRef.current = picked.steps;
              } else {
                setPanelTraceLines(traces);
                setPanelTraceSteps(traceSteps);
                panelTraceLinesRef.current = traces;
                panelTraceStepsRef.current = traceSteps;
              }
            } catch {
              setPanelTraceLines(traces);
              setPanelTraceSteps(traceSteps);
              panelTraceLinesRef.current = traces;
              panelTraceStepsRef.current = traceSteps;
            }
          }
        } catch {
          /* ignore */
        }
      }, 3000);
    }
  };

  useEffect(() => {
    messagesRef.current = messages;
    if (slug && selected?.thread_id && messages.length > 0) {
      setSessionChatCache(slug, selected.thread_id, messages);
    }
  }, [messages, slug, selected?.thread_id]);

  useEffect(
    () => () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
      }
    },
    [],
  );

  const ensureWorkspace = async (): Promise<string | null> => {
    if (slug) {
      return slug;
    }
    if (workspaces.length > 0) {
      const first = workspaces[0].slug;
      setSlug(first);
      return first;
    }
    await alert('请先选择工作区，或点击「打开新工作空间」');
    return null;
  };

  const handleOpenWorkspace = async () => {
    try {
      const picked = await api.pickWorkspaceDirectory('');
      if (picked.cancelled || !picked.path) {
        return;
      }
      const ws = await api.registerWorkspace(picked.path);
      treeFetchSeqRef.current += 1;
      writeStoredWorkspaceMeta({
        slug: ws.slug,
        path: ws.path,
        label: workspaceLabelFromPath(ws.path, ws.slug),
      });
      setSlug(ws.slug);
      setSelected(null);
      setTreeReadySlug(null);
      refreshWorkspaces();
    } catch (err) {
      await alert(err instanceof Error ? err.message : '打开工作区失败');
    }
  };

  const handleDismissWorkspace = async (dismissSlug: string) => {
    const ws = workspaces.find((w) => w.slug === dismissSlug);
    const label = ws?.path.split('/').filter(Boolean).pop() || dismissSlug;
    if (
      !(await confirm({
        title: '移除工作区',
        message: `从最近列表移除「${label}」？\n\n仅隐藏侧栏入口，Agent/Plan 会话数据仍保留；再次打开同一路径可恢复。`,
        confirmLabel: '移除',
      }))
    ) {
      return;
    }
    try {
      const nextList = workspaces.filter((w) => w.slug !== dismissSlug);
      await api.dismissWorkspace(dismissSlug);
      removeRecentWorkspace(dismissSlug);
      if (slug === dismissSlug) {
        const nextSlug = nextList[0]?.slug || '';
        treeFetchSeqRef.current += 1;
        setSlug(nextSlug);
        setSelected(null);
        setCatalogOpen(null);
        setMessages([]);
        setPlanDetail(null);
        setTreeReadySlug(null);
      }
      refreshWorkspaces();
    } catch (err) {
      await alert(err instanceof Error ? err.message : '移除失败');
    }
  };

  const handleNewAgent = async () => {
    const activeSlug = await ensureWorkspace();
    if (!activeSlug) {
      return;
    }
    const { thread_id } = await api.createSession(activeSlug, 'agent');
    refreshTree();
    handleSelect({ kind: 'agent', thread_id, title: '新会话', children: [] });
  };

  const handleNewPlan = async () => {
    const activeSlug = await ensureWorkspace();
    if (!activeSlug) {
      return;
    }
    const goalInput = await prompt({
      title: '新建 Plan',
      message: 'Plan 目标（自然语言）',
      placeholder: '例如：梳理模块依赖并输出重构计划',
      confirmLabel: '创建',
    });
    if (goalInput === null) {
      return;
    }
    const goal = goalInput.trim();
    const { thread_id } = await api.createSession(activeSlug, 'plan', goal);
    refreshTree();
    const node: TreeNode = {
      kind: 'plan',
      thread_id,
      title: goal || thread_id,
      children: [],
    };
    setSelected(node);
    await loadHistory(node);
    if (goal.trim() && selectedRef.current?.thread_id === thread_id) {
      await sendPlanMessage(node, goal);
    }
  };

  const handleCatalogOpen = async (kind: 'skills' | 'rules' | 'tools') => {
    const activeSlug = await ensureWorkspace();
    if (!activeSlug) {
      return;
    }
    setSelected(null);
    setMessages([]);
    setTraceLines([]);
    setTraceSteps([]);
    setPanelTraceLines([]);
    setPanelTraceSteps([]);
    setPlanDetail(null);
    setCatalogOpen(kind);
  };

  const handleDeleteSession = async (node: TreeNode) => {
    if (!slug || busy) {
      return;
    }
    const label =
      node.kind === 'plan'
        ? `Plan「${node.title || node.thread_id}」及其 Worker 节点`
        : `Agent「${node.title || node.thread_id}」`;
    if (
      !(await confirm({
        title: '删除会话',
        message: `确定删除 ${label}？此操作不可恢复。`,
        confirmLabel: '删除',
        danger: true,
      }))
    ) {
      return;
    }
    try {
      abortRef.current?.abort();
      await api.deleteSession(slug, node.thread_id);
      if (isSessionAffectedByDelete(node.thread_id)) {
        clearSessionView();
      }
      refreshTree();
    } catch (err) {
      await alert(err instanceof Error ? err.message : '删除失败');
    }
  };

  const handleRenameSession = async (node: TreeNode, title: string) => {
    if (!slug) {
      throw new Error('未选择工作区');
    }
    const res = await api.renameSession(slug, node.thread_id, title);
    if (selected?.thread_id === node.thread_id) {
      setSelected({ ...selected, title: res.title, title_full: res.title });
    }
    refreshTree();
  };

  const clearSessionView = () => {
    setSelected(null);
    setMessages([]);
    setTraceLines([]);
    setTraceSteps([]);
    setPanelTraceLines([]);
    setPanelTraceSteps([]);
    setPlanDetail(null);
  };

  const isSessionAffectedByDelete = (threadId: string) => {
    if (!selected) {
      return false;
    }
    if (selected.thread_id === threadId) {
      return true;
    }
    return selected.thread_id.startsWith(`${threadId}:`);
  };

  const handleToggleSessionSelect = (threadId: string) => {
    setSelectedSessionIds((prev) => {
      const next = new Set(prev);
      if (next.has(threadId)) {
        next.delete(threadId);
      } else {
        next.add(threadId);
      }
      return next;
    });
  };

  const handleSelectAllSessions = (nodes: TreeNode[]) => {
    setSelectedSessionIds((prev) => {
      const next = new Set(prev);
      const allSelected = nodes.length > 0 && nodes.every((n) => next.has(n.thread_id));
      for (const node of nodes) {
        if (allSelected) {
          next.delete(node.thread_id);
        } else {
          next.add(node.thread_id);
        }
      }
      return next;
    });
  };

  const handleBatchDeleteSessions = async () => {
    if (!slug || busy || selectedSessionIds.size === 0) {
      return;
    }
    const ids = Array.from(selectedSessionIds);
    if (
      !(await confirm({
        title: '批量删除会话',
        message: `确定删除选中的 ${ids.length} 个会话？Plan 将级联删除 Worker。此操作不可恢复。`,
        confirmLabel: '删除',
        danger: true,
      }))
    ) {
      return;
    }
    try {
      abortRef.current?.abort();
      const res = await api.deleteSessions(slug, ids);
      if (ids.some((id) => isSessionAffectedByDelete(id))) {
        clearSessionView();
      }
      setMultiSelectMode(false);
      setSelectedSessionIds(new Set());
      refreshTree();
      if (res.failure_count > 0) {
        const failed = res.results
          .filter((r) => !r.ok)
          .map((r) => `${r.thread_id}: ${r.error || '失败'}`)
          .join('\n');
        await alert(`已删除 ${res.success_count} 个，失败 ${res.failure_count} 个\n${failed}`);
      }
    } catch (err) {
      await alert(err instanceof Error ? err.message : '批量删除失败');
    }
  };

  const persistPanelTraceSnapshot = (threadId: string) => {
    if (!slug || !threadId) {
      return;
    }
    const merged = mergeLiveTraceIntoPanel(
      panelTraceLinesRef.current,
      panelTraceStepsRef.current,
      traceLinesRef.current,
      traceStepsRef.current,
    );
    saveTracePanelCache(slug, threadId, merged.lines, merged.steps);
  };

  const persistPanelTrace = (threadId: string) => {
    if (!slug || !threadId) {
      return;
    }
    saveTracePanelCache(
      slug,
      threadId,
      panelTraceLinesRef.current.map((line) => line.text),
      panelTraceStepsRef.current,
    );
  };

  /** 结束本轮 live trace 区；正文不进聊天列表，仅保留右侧 Trace 面板。 */
  const finalizeLiveTrace = (threadId?: string) => {
    if (threadId) {
      persistPanelTrace(threadId);
    }
    traceFlushedRef.current = true;
    setTraceLines([]);
    setTraceSteps([]);
    setThinkingText('');
    traceLinesRef.current = [];
    traceStepsRef.current = [];
  };

  const {
    maybePromptPlanConfirm,
    handleWorkClick,
    handleBackToPlan,
    handlePlanStop,
    handlePlanAbort,
    handleTaskStop,
    handleTaskRun,
    handlePlanConfirm,
    handlePlanContinue,
  } = usePlanSession({
    slug,
    selected,
    planDetail,
    allowWrite,
    plans,
    alert,
    confirm,
    setBusy,
    setPlanDetail,
    setMessages,
    setAllowWrite,
    setPlanConfirm,
    setTaskStepConfirm,
    taskStepDismissedRef,
    confirmAutoShownRef,
    planStopInFlightRef,
    runningThreadsRef,
    streamAbortRef,
    beginStream,
    bindSSE,
    finalizeLiveTrace,
    handleSelect,
    refreshTree,
  });
  maybePromptPlanConfirmRef.current = maybePromptPlanConfirm;

  handleSSEEventRef.current = createHandleSSEEvent({
    slug,
    selected,
    selectedRef,
    stoppedThreadsRef,
    turnOpenRef,
    runningThreadsRef,
    streamAbortRef,
    streamLastEventAtRef,
    panelTraceLinesRef,
    panelTraceStepsRef,
    panelTraceTurnsRef,
    traceLinesRef,
    traceStepsRef,
    confirmAutoShownRef,
    surveyDismissedRef,
    taskStepDismissedRef,
    traceTurnStartRef,
    streamedRef,
    traceFlushedRef,
    setBusy,
    setThinkingText,
    setTraceActivitySec,
    setTraceLines,
    setTraceSteps,
    setPanelTraceLines,
    setPanelTraceSteps,
    setPanelTraceTurns,
    setStreamText,
    setMessages,
    setPlanDetail,
    setAllowWrite,
    setSurvey,
    setPlanConfirm,
    setTaskStepConfirm,
    setFileChangesTick,
    ensurePlanSubscription,
    maybeReleasePlanSubscription,
    finalizeLiveTrace,
    refreshTree,
    bumpContextRefresh,
  });
  onSSEHandlerRef.current = (event) => ingestSSE(event);

  const liveTraceText = traceLines.map((l) => l.text).join('\n');
  const planViewBusy = busy && selected?.kind === 'plan';
  const mergedPanelTrace = mergeLiveTraceIntoPanel(
    panelTraceLines,
    panelTraceSteps,
    traceLines,
    traceSteps,
  );
  const displayPanelSteps =
    planViewBusy && panelTraceSteps.length > 0
      ? panelTraceSteps
      : mergedPanelTrace.steps.length > 0
        ? mergedPanelTrace.steps
        : panelTraceSteps;
  const displayPanelLines =
    planViewBusy && panelTraceLines.length > 0
      ? panelTraceLines
      : mergedPanelTrace.lines.length > 0
        ? panelLinesFromTexts(mergedPanelTrace.lines)
        : panelTraceLines;
  /** Agent 执行中：对话区仅展示当前轮 live trace；完成后收起 */
  const currentTurnSteps =
    busy && traceSteps.length > 0 ? traceSteps : panelTraceSteps;
  const displayPanelTurns = buildDisplayTraceTurns(panelTraceTurns, currentTurnSteps, {
    busy,
    currentLabel:
      busy && currentTurnSteps.length > 0
        ? `第 ${panelTraceTurns.length + 1} 轮 · 进行中`
        : undefined,
  });
  const agentChatTraceText =
    busy && selected?.kind === 'agent' ? liveTraceText : '';
  const agentChatTraceSteps =
    busy && selected?.kind === 'agent' ? currentTurnSteps : [];
  const agentChatTraceTurns =
    busy && selected?.kind === 'agent' && currentTurnSteps.length > 0
      ? displayPanelTurns.slice(-1)
      : [];
  const planChatTraceText =
    busy && selected?.kind === 'plan'
      ? panelTraceLines.map((l) => l.text).join('\n')
      : busy
        ? liveTraceText
        : '';
  const planChatTraceSteps =
    busy && selected?.kind === 'plan' ? panelTraceSteps : busy ? traceSteps : [];

  const handleModelChange = async (modelId: string) => {
    if (!slug || !modelId || llmSettings?.model === modelId) {
      return;
    }
    const next = await api.setLlmSettings(slug, { model: modelId });
    setLlmSettings(next);
  };

  const handleThinkingChange = async (enabled: boolean) => {
    if (!slug || !llmSettings?.thinking.supported) {
      return;
    }
    if (llmSettings.thinking.enabled === enabled) {
      return;
    }
    try {
      const next = await api.setLlmSettings(slug, { thinking_enabled: enabled });
      setLlmSettings(next);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `think-err-${Date.now()}`,
          role: 'system',
          text: `**Thinking 设置失败:** ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    }
  };

  const handleAllowWriteChange = async (enabled: boolean) => {
    setAllowWrite(enabled);
    bumpContextRefresh();
    if (!slug) {
      return;
    }
    const sel = selectedRef.current;
    if (sel?.kind !== 'agent') {
      api.capabilities(slug, enabled).then(setCaps).catch(() => setCaps(null));
      return;
    }
    try {
      await api.setWriteMode(slug, enabled, sel.thread_id);
      api.capabilities(slug, enabled).then(setCaps).catch(() => setCaps(null));
    } catch (err) {
      setAllowWrite(!enabled);
      bumpContextRefresh();
      setMessages((prev) => [
        ...prev,
        {
          id: `write-mode-err-${Date.now()}`,
          role: 'system',
          text: `**写入模式切换失败:** ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    }
  };

  const handleWebSearchChange = async (enabled: boolean) => {
    if (!slug) {
      return;
    }
    const threadId =
      selected?.kind === 'agent' || selected?.kind === 'plan' ? selected.thread_id : '';
    const res = await api.setWebSearch(slug, enabled, threadId, allowWrite);
    setCaps((prev) => (prev ? { ...prev, web_search_enabled: res.enabled } : prev));
  };

  const handleSandboxChange = async (enabled: boolean) => {
    if (!slug) {
      return;
    }
    const threadId =
      selected?.kind === 'agent' || selected?.kind === 'plan' ? selected.thread_id : '';
    try {
      const res = await api.setSandbox(slug, enabled, threadId, allowWrite);
      setCaps((prev) => (prev && res.sandbox ? { ...prev, sandbox: res.sandbox } : prev));
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `sandbox-err-${Date.now()}`,
          role: 'system',
          text: `**沙箱设置失败:** ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    }
  };

  const handleDeleteEmptySessions = async () => {
    if (!slug) {
      return;
    }
    const res = await api.deleteEmptySessions(slug);
    await alert(res.message);
    refreshTree();
    refreshWorkspaces();
  };

  const handleStop = async () => {
    const tid = selected?.thread_id;
    if (selected?.kind === 'plan' && slug && tid) {
      if (planDetail?.job?.running || busy) {
        void handlePlanStop();
        streamAbortRef.current.get(tid)?.abort();
        streamAbortRef.current.delete(tid);
        finalizeLiveTrace(tid);
        return;
      }
    }
    if (selected?.kind === 'agent' && slug && tid) {
      stoppedThreadsRef.current.add(tid);
      releaseTurnOpen(tid, turnOpenRef.current);
      try {
        await api.abortAgentChat(slug, tid);
      } catch {
        /* ignore */
      }
      setMessages((prev) => [
        ...prev,
        { id: `stop-${Date.now()}`, role: 'system', text: '正在停止当前生成…' },
      ]);
      return;
    }
    if (tid) {
      streamAbortRef.current.get(tid)?.abort();
      streamAbortRef.current.delete(tid);
      runningThreadsRef.current.delete(tid);
    }
    finalizeLiveTrace(tid);
    setBusy(false);
    setStreamText('');
    setThinkingText('');
    setTraceActivitySec(0);
    setMessages((prev) => [
      ...prev,
      { id: `stop-${Date.now()}`, role: 'system', text: '已停止当前生成。' },
    ]);
  };

  const sendAgentMessage = async (
    node: TreeNode,
    text: string,
    images: ComposerImage[] = [],
  ) => {
    if (!slug) {
      return;
    }
    const displayImages = images.map(({ media_type, preview_url, id }) => ({
      media_type,
      url: preview_url,
      id,
    }));
    releaseTurnOpen(node.thread_id, turnOpenRef.current);
    setBusy(true);
    bumpSidebarSession(node, setAgents, setPlans);
    streamedRef.current = false;
    traceFlushedRef.current = false;
    setTraceLines([]);
    setTraceSteps([]);
    traceLinesRef.current = [];
    traceStepsRef.current = [];
    setThinkingText('');
    traceTurnStartRef.current = Date.now();
    setTraceActivitySec(0);
    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: 'user', text, images: displayImages },
    ]);
    requestAnimationFrame(() => mainScroll.stickToBottom());
    savePendingUserMessage(slug, node.thread_id, text);
    streamAbortRef.current.delete(node.thread_id);
    streamLastEventAtRef.current.delete(node.thread_id);
    resubscribeSession(node.thread_id);
    lastSessionEventAtRef.current.set(node.thread_id, Date.now());
    runningThreadsRef.current.add(node.thread_id);
    try {
      await api.startAgentChat(
        slug,
        node.thread_id,
        text,
        allowWrite,
        images.map((img) => img.file),
      );
      refreshTree();
    } catch (e) {
      runningThreadsRef.current.delete(node.thread_id);
      releaseTurnOpen(node.thread_id, turnOpenRef.current);
      void syncRemoteTrace(node.thread_id);
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'system', text: String(e) },
      ]);
      setBusy(false);
    } finally {
      // POST 立即返回时 lock 可能尚未占用，避免误判 running=false 清掉 busy/trace 订阅态
      void syncAgentRunningState(node.thread_id, { updateBusy: false });
    }
  };

  const sendPlanMessage = async (node: TreeNode, text: string) => {
    if (!slug) {
      return;
    }
    setBusy(true);
    streamedRef.current = false;
    traceFlushedRef.current = false;
    setTraceLines([]);
    setTraceSteps([]);
    traceLinesRef.current = [];
    traceStepsRef.current = [];
    setThinkingText('');
    setMessages((prev) => [...prev, { id: `u-${Date.now()}`, role: 'user', text }]);
    requestAnimationFrame(() => mainScroll.stickToBottom());
    const ac = beginStream(node.thread_id);
    try {
      if (planDetail?.phase === 'completed') {
        await api.planDiscuss(slug, node.thread_id, text, bindSSE(node.thread_id), ac.signal);
      } else {
        await api.planStart(slug, node.thread_id, text, allowWrite, bindSSE(node.thread_id), ac.signal);
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        runningThreadsRef.current.delete(node.thread_id);
        streamAbortRef.current.delete(node.thread_id);
        streamLastEventAtRef.current.delete(node.thread_id);
        setBusy(false);
        return;
      }
      streamAbortRef.current.delete(node.thread_id);
      streamLastEventAtRef.current.delete(node.thread_id);
      void syncRemoteTrace(node.thread_id);
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'system', text: String(e) },
      ]);
      setBusy(runningThreadsRef.current.has(node.thread_id));
    } finally {
      if (!runningThreadsRef.current.has(node.thread_id)) {
        streamAbortRef.current.delete(node.thread_id);
        streamLastEventAtRef.current.delete(node.thread_id);
        setBusy(false);
      }
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    const images = composerImages;
    if ((!text && images.length === 0) || !selected || busy) {
      return;
    }
    setInput('');
    setComposerImages([]);
    if (slug && selected) {
      clearComposerDraft(slug, selected.thread_id);
      clearComposerImagesCache(slug, selected.thread_id);
    }

    if (text.startsWith('/') && slug) {
      if (consoleOpsRef.current?.openFromSlash(text)) {
        return;
      }
      setBusy(true);
      try {
        const res = await api.metaCommand(slug, text, allowWrite, selected.thread_id);
        if (res.registered) {
          if (res.display_mode === 'agent') {
            await loadHistory(selected);
            refreshTree();
            bumpContextRefresh();
          } else {
            const title = metaCommandModalTitle(text);
            const body = res.output?.trim() || '（无输出）';
            consoleOpsRef.current?.showMetaOutput(title, body);
            refreshTree();
          }
          return;
        }
      } catch (e) {
        setMessages((prev) => [
          ...prev,
          { id: `err-${Date.now()}`, role: 'system', text: String(e) },
        ]);
        return;
      } finally {
        setBusy(false);
      }
    }

    if (selected.kind === 'agent') {
      await sendAgentMessage(selected, text, images);
    } else if (selected.kind === 'plan') {
      await sendPlanMessage(selected, text);
    }
  };

  const sessionTitle =
    selected?.kind === 'worker'
      ? `Work ${selected.task_id || ''}`
      : selected?.title && selected.title !== selected.thread_id
        ? selected.title
        : selected?.thread_id || 'llgraph';

  const workerMeta = useMemo(() => {
    if (!selected || selected.kind !== 'worker') {
      return null;
    }
    const taskId = selected.task_id || '';
    const planThread = selected.thread_id.split(':worker:')[0];
    const task = planDetail?.tasks.find((t) => String(t.id) === taskId);
    const wfTask = planDetail?.workflow_snapshot?.tasks?.find((t) => t.id === taskId);
    return {
      planThread,
      taskId,
      title: String(task?.title || selected.title || taskId),
      status: String(wfTask?.status || task?.status || selected.status || 'pending'),
      planTitle: planDetail?.title || planThread,
      readonly: Boolean(task?.readonly),
    };
  }, [selected, planDetail]);

  const workerLiveSteps =
    busy && traceSteps.length > 0
      ? mergeTraceStepsUnique(panelTraceSteps, traceSteps)
      : panelTraceSteps;
  const workerLiveText =
    busy && traceLines.length > 0
      ? [...panelTraceLines, ...panelLinesFromTexts(traceLines.map((l) => l.text))]
          .map((l) => l.text)
          .join('\n')
      : panelTraceLines.map((l) => l.text).join('\n');

  const workerTraceText = workerLiveText;

  const canChat = selected && (selected.kind === 'agent' || selected.kind === 'plan');
  const confirmThreadId =
    selected?.kind === 'worker'
      ? selected.thread_id.split(':worker:')[0]
      : selected?.thread_id || '';
  const pendingConfirmCount =
    slug && confirmThreadId ? countPendingConfirms(slug, confirmThreadId) : 0;
  const showPendingSurveyChip =
    Boolean(slug && confirmThreadId && !survey && hasPendingKind(slug, confirmThreadId, 'survey'));
  const showPendingPlanChip =
    Boolean(
      slug && confirmThreadId && !planConfirm && hasPendingKind(slug, confirmThreadId, 'plan_confirm'),
    );
  const pendingTaskStepId = (() => {
    if (!slug || !confirmThreadId || taskStepConfirm) {
      return null;
    }
    const head = peekConfirmHead(slug, confirmThreadId);
    if (head?.kind !== 'task_step_confirm') {
      return null;
    }
    if (taskStepDismissedRef.current === head.id) {
      return String((head.payload as { task_id?: string })?.task_id || '');
    }
    return null;
  })();

  const openPendingConfirm = useCallback(() => {
    if (!slug || !confirmThreadId) {
      return;
    }
    surveyDismissedRef.current = null;
    taskStepDismissedRef.current = null;
    applyPendingConfirmHead(
      slug,
      confirmThreadId,
      { setSurvey, setPlanConfirm, setTaskStepConfirm },
    );
  }, [slug, confirmThreadId, setSurvey, setPlanConfirm, setTaskStepConfirm]);

  const openPendingPlanConfirm = useCallback(() => {
    if (!slug || !confirmThreadId || !planDetail) {
      return;
    }
    ingestPlanConfirmFromDetail(slug, confirmThreadId, planDetail);
    const head = peekConfirmHead(slug, confirmThreadId);
    if (head?.kind === 'plan_confirm') {
      setPlanConfirm(head.payload as Record<string, unknown>);
      return;
    }
    openPendingConfirm();
  }, [slug, confirmThreadId, planDetail, setPlanConfirm, openPendingConfirm]);

  const latestPlanConfirm = useMemo((): PlanConfirmHistoryEntry | null => {
    const history = planDetail?.plan_state?.confirm_history;
    if (!Array.isArray(history) || history.length === 0) {
      return null;
    }
    return history[history.length - 1] as PlanConfirmHistoryEntry;
  }, [planDetail?.plan_state?.confirm_history]);

  const showPlanMain = selected?.kind === 'plan' && planDetail;
  const showWorkerMain = selected?.kind === 'worker' && workerMeta;
  const showRightPanel = Boolean(selected && !catalogOpen && rightPanelOpen);

  const layoutStyle = {
    '--cursor-sidebar-width': `${sidebarWidth}px`,
    '--cursor-right-panel-width': `${rightPanelWidth}px`,
  } as React.CSSProperties;

  const sidebarAgents = agents;
  const sidebarPlans = plans;
  const sessionTreeLoading = Boolean(slug && treeLoading && treeReadySlug !== slug);

  return (
    <div
      className={`cursor-layout${
        showRightPanel ? '' : ' cursor-layout--right-collapsed'
      }${panelResizing ? ' cursor-layout--resizing' : ''}`}
      style={layoutStyle}
    >
      <CursorSidebar
        slug={slug}
        workspaces={displayWorkspaces}
        workspaceDisplay={workspaceDisplay}
        workspacesLoading={workspacesLoading}
        agents={sidebarAgents}
        plans={sidebarPlans}
        treeLoading={sessionTreeLoading}
        selectedId={selected?.thread_id || null}
        catalogOpen={catalogOpen}
        multiSelectMode={multiSelectMode}
        selectedSessionIds={selectedSessionIds}
        busy={busy}
        onSlugChange={(s) => {
          if (s === slug) {
            refreshTree();
            return;
          }
          treeFetchSeqRef.current += 1;
          setSlug(s);
          setSelected(null);
          setCatalogOpen(null);
          setMultiSelectMode(false);
          setSelectedSessionIds(new Set());
          setTreeReadySlug(null);
          setAgents([]);
          setPlans([]);
          setMessages([]);
          setPlanDetail(null);
          if (s) {
            void api.touchWorkspace(s).then(() => refreshWorkspaces()).catch(() => {});
          }
        }}
        onOpenWorkspace={handleOpenWorkspace}
        onDismissWorkspace={(s) => void handleDismissWorkspace(s)}
        onSelect={handleSelect}
        onNewAgent={handleNewAgent}
        onNewPlan={handleNewPlan}
        onDelete={handleDeleteSession}
        onRename={handleRenameSession}
        onCatalogOpen={handleCatalogOpen}
        onEnterMultiSelect={() => setMultiSelectMode(true)}
        onExitMultiSelect={() => {
          setMultiSelectMode(false);
          setSelectedSessionIds(new Set());
        }}
        onToggleSessionSelect={handleToggleSessionSelect}
        onSelectAllSessions={handleSelectAllSessions}
        onBatchDelete={handleBatchDeleteSessions}
        onDeleteEmpty={handleDeleteEmptySessions}
        onCodeSearch={() => setCodeSearchOpen(true)}
      />

      <main className="cursor-main">
        <PanelResizeHandle
          edge="left"
          active={panelResizing === 'sidebar'}
          title="调整左侧会话栏宽度"
          onMouseDown={(e) => {
            e.preventDefault();
            setPanelResizing('sidebar');
          }}
        />
        {showRightPanel && (
          <PanelResizeHandle
            edge="right"
            active={panelResizing === 'right'}
            title="调整右侧面板宽度"
            onMouseDown={(e) => {
              e.preventDefault();
              setPanelResizing('right');
            }}
          />
        )}
        {catalogOpen ? (
          <CatalogPanel
            slug={slug}
            kind={catalogOpen}
            caps={caps}
            onClose={() => setCatalogOpen(null)}
            onCapsRefresh={refreshCaps}
          />
        ) : !selected ? (
          <div className="cursor-welcome">
            <h1>llgraph Agent</h1>
            <p>选择左侧工作区，创建 Agent / Plan，或浏览 Skills / Rules / 工具</p>
          </div>
        ) : (
          <>
            <header className="cursor-main-header">
              <div className="cursor-main-header-main">
                <EditableSessionTitle
                  title={sessionTitle}
                  titleFull={selected.title_full}
                  threadId={selected.thread_id}
                  renamable={selected.kind === 'agent' || selected.kind === 'plan'}
                  onRename={
                    selected.kind === 'agent' || selected.kind === 'plan'
                      ? (title) => handleRenameSession(selected, title)
                      : undefined
                  }
                />
                {(selected.kind === 'agent' || selected.kind === 'plan') && (
                  <AgentToolbar
                    llm={llmSettings}
                    busy={busy}
                    isAgent={selected.kind === 'agent'}
                    webSearchEnabled={caps?.web_search_enabled ?? false}
                    sandbox={caps?.sandbox}
                    allowWrite={allowWrite}
                    onAllowWriteChange={(v) => void handleAllowWriteChange(v)}
                    onSandboxChange={handleSandboxChange}
                    onModelChange={handleModelChange}
                    onThinkingChange={handleThinkingChange}
                    onWebSearchChange={
                      selected.kind === 'agent' || selected.kind === 'plan'
                        ? handleWebSearchChange
                        : undefined
                    }
                  />
                )}
              </div>
              <div className="cursor-main-actions">
                {selected && (selected.kind === 'agent' || selected.kind === 'plan') && (
                  <ConsoleOps
                    ref={consoleOpsRef}
                    slug={slug}
                    threadId={selected.thread_id}
                    isAgent={selected.kind === 'agent'}
                    allowWrite={allowWrite}
                    busy={busy}
                    contextRefreshSignal={contextRefreshSignal}
                    sandbox={caps?.sandbox}
                    onOpenLog={() => {
                      setRightPanelOpen(true);
                      setRightPanelRequestedTab('log');
                    }}
                  />
                )}
                {selected && (selected.kind === 'agent' || selected.kind === 'plan') && (
                  <button
                    type="button"
                    className="cursor-panel-toggle"
                    onClick={() => setRightPanelOpen((v) => !v)}
                    title={rightPanelOpen ? '收起右侧面板' : '展开右侧面板'}
                    aria-expanded={rightPanelOpen}
                  >
                    {rightPanelOpen ? '面板 ▸' : '◂ 面板'}
                  </button>
                )}
              </div>
            </header>

            <div className="cursor-main-scroll" ref={mainScroll.ref}>
              {selected?.kind === 'plan' && !planDetail ? (
                <div className="cursor-catalog-empty">加载 Plan 工作流…</div>
              ) : showPlanMain ? (
                <>
                  <PlanMainPanel
                    slug={slug}
                    planDetail={planDetail}
                    busy={busy}
                    onTaskSelect={handleWorkClick}
                    onPlanConfirm={openPendingPlanConfirm}
                    onPlanContinue={handlePlanContinue}
                    onPlanStop={() => void handlePlanStop()}
                    onPlanAbort={() => void handlePlanAbort()}
                    onTaskStop={(taskId) => void handleTaskStop(taskId)}
                    onTaskRun={(taskId) => void handleTaskRun(taskId)}
                  />
                  <section className="cursor-plan-chat">
                    <header className="cursor-plan-chat-header">
                      <h2 className="cursor-plan-chat-title">对话与修订</h2>
                      <p className="cursor-plan-chat-hint">
                        目标与修订说明在此；Planner 调研过程见右侧 Trace 面板
                      </p>
                      {showPendingPlanChip && (
                        <button
                          type="button"
                          className="plan-confirm-summary-chip"
                          onClick={openPendingConfirm}
                        >
                          待确认计划
                        </button>
                      )}
                      {latestPlanConfirm && !planConfirm && !showPendingPlanChip && (
                        <PlanConfirmSummaryChip
                          entry={latestPlanConfirm}
                          onClick={() => setPlanConfirmReview(latestPlanConfirm)}
                        />
                      )}
                    </header>
                    <ChatThread
                      messages={messages}
                      liveTraceText={planChatTraceText}
                      liveTraceSteps={planChatTraceSteps}
                      streamText={streamText}
                      busy={busy}
                      historyLoading={historyLoading}
                      traceMode={caps?.trace_mode}
                    />
                  </section>
                </>
              ) : showWorkerMain ? (
                <WorkerMainPanel
                  planTitle={workerMeta.planTitle}
                  taskId={workerMeta.taskId}
                  taskTitle={workerMeta.title}
                  taskStatus={workerMeta.status}
                  taskReadonly={workerMeta.readonly}
                  messages={messages}
                  traceLines={workerTraceText}
                  liveTraceSteps={workerLiveSteps}
                  busy={busy || workerMeta.status === 'running'}
                  onBack={handleBackToPlan}
                  onStop={() => void handleTaskStop(workerMeta.taskId)}
                  onRun={() => void handleTaskRun(workerMeta.taskId)}
                />
              ) : (
                <>
                  {(showPendingSurveyChip || showPendingPlanChip || pendingTaskStepId) && (
                    <div className="pending-confirm-bar">
                      {showPendingSurveyChip && (
                        <button
                          type="button"
                          className="plan-confirm-summary-chip"
                          onClick={openPendingConfirm}
                        >
                          有待确认问卷{pendingConfirmCount > 1 ? ` (${pendingConfirmCount})` : ''}
                        </button>
                      )}
                      {showPendingPlanChip && (
                        <button
                          type="button"
                          className="plan-confirm-summary-chip"
                          onClick={openPendingConfirm}
                        >
                          待确认计划
                        </button>
                      )}
                      {pendingTaskStepId && (
                        <div className="pending-confirm-task-step">
                          <span>Plan 等待继续执行 Work {pendingTaskStepId}</span>
                          <button type="button" className="cursor-btn-primary" onClick={() => void handlePlanContinue()}>
                            继续执行
                          </button>
                          <button type="button" onClick={openPendingConfirm}>
                            打开确认框
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                  <ChatThread
                  messages={messages}
                  liveTraceText={agentChatTraceText}
                  liveTraceSteps={agentChatTraceSteps}
                  liveTraceTurns={agentChatTraceTurns}
                  streamText={streamText}
                  busy={busy}
                  historyLoading={historyLoading}
                  traceMode={caps?.trace_mode}
                />
                </>
              )}
              <div ref={chatEndRef} />
            </div>

            {fileChangesConfig && !canChat && (
              <FileChangesPanel
                key={`${fileChangesConfig.mode}-${fileChangesConfig.sessionThreadId}-${fileChangesTick}`}
                slug={slug}
                mode={fileChangesConfig.mode}
                sessionThreadId={fileChangesConfig.sessionThreadId}
                planThreadId={fileChangesConfig.planThreadId}
                taskId={fileChangesConfig.taskId}
                busy={busy}
                allowWrite={allowWrite}
                onChangesUpdated={() => setFileChangesTick((n) => n + 1)}
              />
            )}

            {canChat ? (
              fileChangesConfig ? (
                <ComposerDock
                  fileChanges={{
                    slug,
                    mode: fileChangesConfig.mode,
                    sessionThreadId: fileChangesConfig.sessionThreadId,
                    planThreadId: fileChangesConfig.planThreadId,
                    taskId: fileChangesConfig.taskId,
                    refreshKey: fileChangesTick,
                    onChangesUpdated: () => setFileChangesTick((n) => n + 1),
                  }}
                  value={input}
                  onChange={setInput}
                  images={composerImages}
                  onImagesChange={setComposerImages}
                  onSend={handleSend}
                  onStop={handleStop}
                  busy={busy}
                  disabled={busy}
                  slashCatalog={slashCatalog}
                  placeholder={
                    selected.kind === 'plan' ? 'Plan 目标或修订说明…' : 'Ask llgraph…'
                  }
                />
              ) : (
                <ChatComposer
                  value={input}
                  onChange={setInput}
                  images={composerImages}
                  onImagesChange={setComposerImages}
                  onSend={handleSend}
                  onStop={handleStop}
                  busy={busy}
                  disabled={busy}
                  slashCatalog={slashCatalog}
                  placeholder={
                    selected.kind === 'plan' ? 'Plan 目标或修订说明…' : 'Ask llgraph…'
                  }
                />
              )
            ) : (
              <div className="cursor-composer-wrap cursor-composer-readonly">
                Worker 只读 trace · 由 Plan 自动执行
              </div>
            )}
          </>
        )}
      </main>

      {selected && !catalogOpen && (
        <>
          {!rightPanelOpen && (
            <button
              type="button"
              className="cursor-panel-edge cursor-panel-edge--expand"
              onClick={() => setRightPanelOpen(true)}
              title="展开 Trace / Tools 面板"
              aria-label="展开右侧面板"
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M10 4L6 8l4 4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
          <div className={`cursor-right-wrap${rightPanelOpen ? '' : ' is-collapsed'}`}>
            <button
              type="button"
              className="cursor-panel-edge cursor-panel-edge--collapse"
              onClick={() => setRightPanelOpen(false)}
              title="收起面板"
              aria-label="收起右侧面板"
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <CursorRightPanel
              caps={caps}
              traceLines={displayPanelLines}
              traceSteps={displayPanelSteps}
              traceTurns={displayPanelTurns}
              liveThinking={thinkingText}
              planDetail={selected.kind === 'plan' ? planDetail : null}
              slug={slug}
              threadId={selected.thread_id}
              isPlan={selected.kind === 'plan'}
              isAgent={selected.kind === 'agent'}
              allowWrite={allowWrite}
              requestedTab={rightPanelRequestedTab}
              onRequestedTabHandled={() => setRightPanelRequestedTab(null)}
              onTraceMode={(m) => {
                if (!slug) {
                  return;
                }
                api.setTraceMode(slug, m).then((r) => {
                  setCaps((prev) => (prev ? { ...prev, trace_mode: r.mode } : prev));
                });
              }}
              onPlanConfirm={openPendingPlanConfirm}
              onPlanContinue={handlePlanContinue}
              busy={busy}
              traceActivitySec={traceActivitySec}
              contextRefreshSignal={contextRefreshSignal}
              onTaskSelect={handleWorkClick}
              onCapsLoaded={setCaps}
            />
          </div>
        </>
      )}

      {survey && slug && (
        <SurveyDialog
          survey={survey}
          onCancel={() => {
            if (slug && confirmThreadId) {
              const head = peekConfirmHead(slug, confirmThreadId);
              if (head?.kind === 'survey') {
                surveyDismissedRef.current = head.id;
              }
            }
            setSurvey(null);
          }}
          onSubmit={async (answers) => {
            if (slug && confirmThreadId) {
              const head = peekConfirmHead(slug, confirmThreadId);
              if (head?.kind === 'survey') {
                dequeueConfirmHead(slug, confirmThreadId);
                applyPendingConfirmHead(
                  slug,
                  confirmThreadId,
                  { setSurvey, setPlanConfirm, setTaskStepConfirm },
                );
              }
            }
            surveyDismissedRef.current = null;
            setSurvey(null);
            try {
              const { message } = await api.formatSurveyAnswers(slug, answers, allowWrite);
              if (selected?.kind === 'agent') {
                sendAgentMessage(selected, message);
              } else if (selected?.kind === 'plan') {
                sendPlanMessage(selected, message);
              }
            } catch (err) {
              await alert(err instanceof Error ? err.message : String(err));
            }
          }}
        />
      )}

      {taskStepConfirm && (
        <TaskStepConfirmDialog
          taskId={taskStepConfirm}
          onDismiss={() => {
            if (slug && confirmThreadId) {
              const head = peekConfirmHead(slug, confirmThreadId);
              if (head?.kind === 'task_step_confirm') {
                taskStepDismissedRef.current = head.id;
              }
            }
            setTaskStepConfirm(null);
          }}
          onContinue={() => {
            setTaskStepConfirm(null);
            handlePlanContinue();
          }}
        />
      )}

      {planConfirm && (
        <PlanConfirmDialog
          payload={planConfirm}
          onCancel={() => {
            setPlanConfirm(null);
          }}
          onConfirm={handlePlanConfirm}
        />
      )}

      {planConfirmReview && (
        <PlanConfirmReviewDialog
          entry={planConfirmReview}
          onClose={() => setPlanConfirmReview(null)}
        />
      )}

      {codeSearchOpen && slug && (
        <CodeSearchPanel slug={slug} onClose={() => setCodeSearchOpen(false)} />
      )}
    </div>
  );
}
