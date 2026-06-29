import { useCallback, useRef, useState } from 'react';
import { api, type MessageItem, type PlanDetail, type TreeNode } from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import type { TraceStep } from '../types/trace';
import { stepsToPanelLogLines } from '../types/trace';
import {
  clearPendingUserMessage,
  loadPendingUserMessage,
} from '../utils/pendingUserMessage';
import { loadTracePanelCache } from '../utils/tracePanelStore';
import { getSessionChatCache, setSessionChatCache } from '../utils/sessionChatCache';
import {
  dedupeConsecutiveUserMessages,
  dedupeUserMessages,
  mergeChatWithPendingUserMessages,
  parseAgentHistoryMessages,
  mergeRunningSessionMessages,
  userMessageAlreadyInChat,
} from '../utils/messageText';
import { buildPlanChatMessages } from '../utils/planChat';
import { planPlannerSubThread, planPlannerVersion } from '../utils/planPlannerTrace';
import type { TraceLine } from '../pages/console/types';
import { planExecutionAllowWrite } from '../pages/console/planHelpers';
import {
  loadTraceTurnsFromRemote,
  panelLinesFromTexts,
  parseTraceStep,
  parseTraceTurnsFromRemote,
  preferRicherTraceCache,
  releaseTurnOpen,
  restorePanelTraceFromMessages,
} from '../pages/console/traceUtils';
import { restorePendingConfirmsFromHistory } from '../utils/pendingConfirmUi';
import { parseWorkerMessages } from '../pages/console/workerUtils';

export type SessionHistoryDeps = {
  slug: string;
  maybePromptPlanConfirm: (detail: PlanDetail, threadId: string) => void;
  ensurePlanSubscription: (planThread: string) => void;
  ensurePlanPlannerSubscription: (planThread: string, versionOverride?: number) => string;
  ensureSessionSubscription: (sessionThread: string) => void;
  resumeSessionAfterSelect: (node: TreeNode) => Promise<void>;
  syncAgentRunningState: (threadId: string, opts?: { updateBusy?: boolean }) => Promise<boolean>;
  runningThreadsRef: React.MutableRefObject<Set<string>>;
  turnOpenRef: React.MutableRefObject<Set<string>>;
  traceLinesRef: React.MutableRefObject<TraceLine[]>;
  traceStepsRef: React.MutableRefObject<TraceStep[]>;
  panelTraceLinesRef: React.MutableRefObject<TraceLine[]>;
  panelTraceStepsRef: React.MutableRefObject<TraceStep[]>;
  panelTraceTurnsRef: React.MutableRefObject<import('../types/trace').TraceTurn[]>;
  surveyDismissedRef: React.MutableRefObject<string | null>;
  taskStepDismissedRef: React.MutableRefObject<string | null>;
  streamAbortRef: React.MutableRefObject<Map<string, AbortController>>;
  setAllowWrite: React.Dispatch<React.SetStateAction<boolean>>;
  setCaps: React.Dispatch<React.SetStateAction<import('../api/client').Capabilities | null>>;
  setPlanDetail: React.Dispatch<React.SetStateAction<PlanDetail | null>>;
  setSurvey: React.Dispatch<React.SetStateAction<import('../api/client').SurveySpec | null>>;
  setPlanConfirm: React.Dispatch<React.SetStateAction<Record<string, unknown> | null>>;
  setTaskStepConfirm: React.Dispatch<React.SetStateAction<string | null>>;
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  setTraceLines: React.Dispatch<React.SetStateAction<TraceLine[]>>;
  setTraceSteps: React.Dispatch<React.SetStateAction<TraceStep[]>>;
  setStreamText: React.Dispatch<React.SetStateAction<string>>;
  setThinkingText: React.Dispatch<React.SetStateAction<string>>;
  setPanelTraceLines: React.Dispatch<React.SetStateAction<TraceLine[]>>;
  setPanelTraceSteps: React.Dispatch<React.SetStateAction<TraceStep[]>>;
  setPanelTraceTurns: React.Dispatch<React.SetStateAction<import('../types/trace').TraceTurn[]>>;
  setBusy: React.Dispatch<React.SetStateAction<boolean>>;
  bumpContextRefresh: () => void;
};

export function useSessionHistory(deps: SessionHistoryDeps) {
  const loadSeqRef = useRef(0);
  const [historyLoading, setHistoryLoading] = useState(false);

  const {
    slug,
    maybePromptPlanConfirm,
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
  } = deps;

  const loadHistory = useCallback(
    async (node: TreeNode) => {
      if (!slug) {
        return;
      }
      const seq = ++loadSeqRef.current;
      const isCurrent = () => loadSeqRef.current === seq;
      setHistoryLoading(true);

      try {
      if (node.kind === 'agent') {
        let sessionAllowWrite = false;
        try {
          const sessionInfo = await api.sessionMeta(slug, node.thread_id);
          sessionAllowWrite = Boolean(sessionInfo.allow_write);
        } catch {
          /* 无 meta 时默认只读 */
        }
        if (!isCurrent()) {
          return;
        }
        setAllowWrite(sessionAllowWrite);
        api.capabilities(slug, sessionAllowWrite).then(setCaps).catch(() => setCaps(null));

        let data: { messages: MessageItem[] };
        try {
          data = await api.messages(slug, node.thread_id);
        } catch (err) {
          if (!isCurrent()) {
            return;
          }
          setMessages([
            {
              id: 'load-err',
              role: 'system',
              text: `**加载历史失败:** ${err instanceof Error ? err.message : String(err)}`,
            },
          ]);
          return;
        }
        if (!isCurrent()) {
          return;
        }
        const parsed = parseAgentHistoryMessages(data.messages);
        let panelLines: TraceLine[] = [];
        let panelSteps: TraceStep[] = [];
        let panelTurns: import('../types/trace').TraceTurn[] = [];
        try {
          const remote = await api.lastTrace(slug, node.thread_id);
          const loaded = loadTraceTurnsFromRemote(remote);
          panelTurns = loaded.completed;
          panelSteps = loaded.currentSteps;
          if (remote.log_lines?.length) {
            panelLines = panelLinesFromTexts(remote.log_lines);
          } else if (panelSteps.length > 0) {
            panelLines = stepsToPanelLogLines(panelSteps);
          }
        } catch {
          /* 无落盘 trace 时走缓存 */
        }
        const picked = preferRicherTraceCache(slug, node.thread_id, panelLines, panelSteps);
        panelLines = picked.lines;
        panelSteps = picked.steps;
        const sessionRunning =
          runningThreadsRef.current.has(node.thread_id) ||
          (await syncAgentRunningState(node.thread_id, { updateBusy: false }));
        if (!isCurrent()) {
          return;
        }
        const pendingUser = sessionRunning ? loadPendingUserMessage(slug, node.thread_id) : null;
        if (!sessionRunning) {
          clearPendingUserMessage(slug, node.thread_id);
          releaseTurnOpen(node.thread_id, turnOpenRef.current);
        }
        const chatMerged = dedupeUserMessages(
          dedupeConsecutiveUserMessages(
            mergeChatWithPendingUserMessages(parsed, {
              traceLines: sessionRunning ? panelLines.map((l) => l.text) : [],
              pendingText: pendingUser,
              allowTraceUser: sessionRunning,
            }),
          ),
        );
        if (pendingUser && userMessageAlreadyInChat(chatMerged, pendingUser)) {
          clearPendingUserMessage(slug, node.thread_id);
        }
        setPlanDetail(null);
        setSurvey(null);
        setPlanConfirm(null);
        setTaskStepConfirm(null);
        surveyDismissedRef.current = null;
        taskStepDismissedRef.current = null;
        setTraceLines([]);
        setTraceSteps([]);
        setStreamText('');
        setThinkingText('');
        traceLinesRef.current = [];
        traceStepsRef.current = [];
        if (sessionRunning) {
          const cached = getSessionChatCache(slug, node.thread_id);
          const finalMessages = mergeRunningSessionMessages(cached, chatMerged);
          setMessages(finalMessages);
          setSessionChatCache(slug, node.thread_id, finalMessages);
        } else {
          setMessages(chatMerged);
          setSessionChatCache(slug, node.thread_id, chatMerged);
        }
        bumpContextRefresh();
        if (panelLines.length === 0 && panelSteps.length === 0) {
          const cached = loadTracePanelCache(slug, node.thread_id);
          if (cached) {
            panelLines = panelLinesFromTexts(cached.log_lines);
            panelSteps = cached.steps;
          } else {
            const fromChat = restorePanelTraceFromMessages(parsed);
            panelLines = fromChat.lines;
            panelSteps = fromChat.steps;
          }
        }
        if (panelTurns.length === 0 && panelSteps.length > 0) {
          panelTurns = parseTraceTurnsFromRemote(undefined, panelSteps);
          panelSteps = [];
        }
        setPanelTraceLines(panelLines);
        setPanelTraceSteps(panelSteps);
        setPanelTraceTurns(panelTurns);
        panelTraceLinesRef.current = panelLines;
        panelTraceStepsRef.current = panelSteps;
        panelTraceTurnsRef.current = panelTurns;
        if (sessionRunning) {
          runningThreadsRef.current.add(node.thread_id);
          turnOpenRef.current.add(node.thread_id);
        }
        setBusy(sessionRunning);
        ensureSessionSubscription(node.thread_id);
        await restorePendingConfirmsFromHistory({
          slug,
          threadId: node.thread_id,
          messages: sessionRunning
            ? mergeRunningSessionMessages(
                getSessionChatCache(slug, node.thread_id),
                chatMerged,
              )
            : chatMerged,
          setters: { setSurvey, setPlanConfirm, setTaskStepConfirm },
          surveyDismissedRef,
          taskStepDismissedRef,
        });
        void resumeSessionAfterSelect(node);
      } else if (node.kind === 'plan') {
        let detail: Awaited<ReturnType<typeof api.plan>>;
        try {
          detail = await api.plan(slug, node.thread_id);
        } catch (err) {
          if (!isCurrent()) {
            return;
          }
          setMessages([
            {
              id: 'plan-load-err',
              role: 'system',
              text: `**加载 Plan 失败:** ${err instanceof Error ? err.message : String(err)}`,
            },
          ]);
          setBusy(false);
          return;
        }
        if (!isCurrent()) {
          return;
        }
        if (detail.job?.running) {
          runningThreadsRef.current.add(node.thread_id);
          ensurePlanSubscription(node.thread_id);
        }
        maybePromptPlanConfirm(detail, node.thread_id);
        let history: MessageItem[] = [];
        try {
          const data = await api.messages(slug, node.thread_id);
          history = data.messages || [];
        } catch {
          /* plan 会话可能尚无 messages.jsonl */
        }
        if (!isCurrent()) {
          return;
        }
        setPlanDetail(detail);
        setSurvey(null);
        setPlanConfirm(null);
        setTaskStepConfirm(null);
        surveyDismissedRef.current = null;
        taskStepDismissedRef.current = null;
        if (planExecutionAllowWrite(detail)) {
          setAllowWrite(true);
        } else {
          setAllowWrite(false);
        }
        setTraceLines([]);
        setTraceSteps([]);
        setStreamText('');
        setThinkingText('');
        traceLinesRef.current = [];
        traceStepsRef.current = [];
        setMessages(buildPlanChatMessages(detail, history));
        let panelLines: TraceLine[] = [];
        let panelSteps: TraceStep[] = [];
        const plannerVer = planPlannerVersion(detail);
        const plannerThread = planPlannerSubThread(node.thread_id, plannerVer);
        try {
          const remote = await api.lastTrace(slug, plannerThread);
          if (remote.steps?.length) {
            panelSteps = remote.steps.map((s) => parseTraceStep(s));
          }
          if (remote.log_lines?.length) {
            panelLines = panelLinesFromTexts(remote.log_lines);
          } else if (panelSteps.length > 0) {
            panelLines = stepsToPanelLogLines(panelSteps);
          }
        } catch {
          /* Planner trace 可能尚未落盘 */
        }
        const pickedPlanner = preferRicherTraceCache(slug, plannerThread, panelLines, panelSteps);
        panelLines = pickedPlanner.lines;
        panelSteps = pickedPlanner.steps;
        if (panelLines.length === 0 && panelSteps.length === 0) {
          try {
            const remote = await api.lastTrace(slug, node.thread_id);
            if (remote.steps?.length) {
              panelSteps = remote.steps.map((s) => parseTraceStep(s));
            }
            if (remote.log_lines?.length) {
              panelLines = panelLinesFromTexts(remote.log_lines);
            } else if (panelSteps.length > 0) {
              panelLines = stepsToPanelLogLines(panelSteps);
            }
          } catch {
            /* ignore */
          }
          const picked = preferRicherTraceCache(slug, node.thread_id, panelLines, panelSteps);
          panelLines = picked.lines;
          panelSteps = picked.steps;
        }
        if (panelLines.length === 0 && panelSteps.length === 0) {
          const cached = loadTracePanelCache(slug, plannerThread) || loadTracePanelCache(slug, node.thread_id);
          if (cached) {
            panelLines = panelLinesFromTexts(cached.log_lines);
            panelSteps = cached.steps;
          }
        }
        ensurePlanPlannerSubscription(node.thread_id);
        const planRunning = Boolean(detail.job?.running);
        if (!planRunning) {
          runningThreadsRef.current.delete(node.thread_id);
          streamAbortRef.current.delete(node.thread_id);
        }
        if (!isCurrent()) {
          return;
        }
        setPanelTraceLines(panelLines);
        setPanelTraceSteps(panelSteps);
        panelTraceLinesRef.current = panelLines;
        panelTraceStepsRef.current = panelSteps;
        setBusy(planRunning);
        const planMessages = buildPlanChatMessages(detail, history);
        await restorePendingConfirmsFromHistory({
          slug,
          threadId: node.thread_id,
          messages: planMessages,
          planDetail: detail,
          setters: { setSurvey, setPlanConfirm, setTaskStepConfirm },
          surveyDismissedRef,
          taskStepDismissedRef,
        });
        void resumeSessionAfterSelect(node);
      } else if (node.kind === 'worker') {
        const planThread = node.thread_id.split(':worker:')[0];
        const taskId = node.task_id || '';
        ensureSessionSubscription(node.thread_id);
        const [data, detail] = await Promise.all([
          api.worker(slug, planThread, taskId),
          api.plan(slug, planThread),
        ]);
        if (!isCurrent()) {
          return;
        }
        setPlanDetail(detail);
        setSurvey(null);
        setPlanConfirm(null);
        setTaskStepConfirm(null);
        surveyDismissedRef.current = null;
        taskStepDismissedRef.current = null;
        const { chat, traces, traceSteps } = parseWorkerMessages(data);
        setTraceLines([]);
        setTraceSteps([]);
        setStreamText('');
        setThinkingText('');
        traceLinesRef.current = [];
        traceStepsRef.current = [];
        setMessages(chat);
        let panelLines = traces;
        let panelSteps = traceSteps;
        try {
          const remote = await api.lastTrace(slug, node.thread_id);
          if (remote.steps?.length) {
            panelSteps = remote.steps.map((s) => parseTraceStep(s));
          }
          if (remote.log_lines?.length) {
            panelLines = panelLinesFromTexts(remote.log_lines);
          } else if (panelSteps.length > 0) {
            panelLines = stepsToPanelLogLines(panelSteps);
          }
        } catch {
          /* 无子 Agent trace 落盘 */
        }
        const pickedWorker = preferRicherTraceCache(slug, node.thread_id, panelLines, panelSteps);
        panelLines = pickedWorker.lines;
        panelSteps = pickedWorker.steps;
        if (panelLines.length === 0 && panelSteps.length === 0) {
          const cached = loadTracePanelCache(slug, node.thread_id);
          if (cached) {
            panelLines = panelLinesFromTexts(cached.log_lines);
            panelSteps = cached.steps;
          }
        }
        if (!isCurrent()) {
          return;
        }
        setTraceLines([]);
        setTraceSteps([]);
        setPanelTraceLines(panelLines);
        setPanelTraceSteps(panelSteps);
        panelTraceLinesRef.current = panelLines;
        panelTraceStepsRef.current = panelSteps;
        const planRunning =
          runningThreadsRef.current.has(planThread) || Boolean(detail.job?.running);
        setBusy(planRunning);
        ensureSessionSubscription(node.thread_id);
        await restorePendingConfirmsFromHistory({
          slug,
          threadId: planThread,
          planDetail: detail,
          setters: { setSurvey, setPlanConfirm, setTaskStepConfirm },
          surveyDismissedRef,
          taskStepDismissedRef,
        });
        void resumeSessionAfterSelect(node);
      }
      } finally {
        if (isCurrent()) {
          setHistoryLoading(false);
        }
      }
    },
    [slug, maybePromptPlanConfirm, ensurePlanSubscription, ensurePlanPlannerSubscription, ensureSessionSubscription, resumeSessionAfterSelect, syncAgentRunningState, bumpContextRefresh],
  );

  return { loadHistory, historyLoading, loadSeqRef };
}
