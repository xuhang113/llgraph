import { useCallback, useRef, useState } from 'react';
import { api, type MessageItem, type TreeNode } from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import type { TraceStep } from '../types/trace';
import { mergeTraceStepsUnique, stepsToPanelLogLines } from '../types/trace';
import {
  clearPendingUserMessage,
  loadPendingUserMessage,
} from '../utils/pendingUserMessage';
import { loadTracePanelCache } from '../utils/tracePanelStore';
import { getSessionChatCache, setSessionChatCache } from '../utils/sessionChatCache';
import { hydrateChatMessageImages } from '../utils/chatAttachmentUrl';
import { preferRicherAgentChat } from '../utils/fetchEnrichedAgentChat';
import {
  dedupeConsecutiveUserMessages,
  dedupeUserMessages,
  finalizeAgentHistoryChat,
  parseAgentHistoryMessages,
  mergeRunningSessionMessages,
  userMessageAlreadyInChat,
} from '../utils/messageText';
import type { TraceLine } from '../pages/console/types';
import { readStoredAllowWrite, readStoredSandboxEnabled } from '../pages/console/storage';
import {
  loadTraceTurnsFromRemote,
  panelLinesFromTexts,
  parseTraceSteps,
  parseTraceTurnsFromRemote,
  preferRicherTraceCache,
  releaseTurnOpen,
  restorePanelTraceFromMessages,
} from '../pages/console/traceUtils';
export type SessionHistoryDeps = {
  slug: string;
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
  setAllowWrite: React.Dispatch<React.SetStateAction<boolean>>;
  setCaps: React.Dispatch<React.SetStateAction<import('../api/client').Capabilities | null>>;
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
    setAllowWrite,
    setCaps,
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
      if (node.kind === 'agent' || node.kind === 'subagent') {
        const sessionAllowWrite =
          node.kind === 'subagent' ? false : readStoredAllowWrite();
        if (!isCurrent()) {
          return;
        }
        setAllowWrite(sessionAllowWrite);
        if (sessionAllowWrite) {
          void api.setWriteMode(slug, true, node.thread_id).catch(() => {});
        }
        void api
          .capabilities(slug, sessionAllowWrite)
          .then(async (capsData) => {
            if (!isCurrent()) {
              return;
            }
            setCaps(capsData);
            if (
              node.kind === 'agent' &&
              readStoredSandboxEnabled() &&
              capsData.sandbox &&
              !capsData.sandbox.enabled &&
              capsData.sandbox.cli_override !== false
            ) {
              try {
                const res = await api.setSandbox(slug, true, node.thread_id, sessionAllowWrite);
                if (isCurrent() && res.sandbox) {
                  setCaps((prev) => (prev ? { ...prev, sandbox: res.sandbox } : prev));
                }
              } catch {
                /* 沙箱后端不可用时保持未勾选 */
              }
            }
          })
          .catch(() => setCaps(null));

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
        let enrichSteps: TraceStep[] = [];
        try {
          const remote = await api.sessionTrace(slug, node.thread_id);
          const loaded = loadTraceTurnsFromRemote(remote);
          panelTurns = loaded.completed;
          panelSteps = loaded.currentSteps;
          if (remote.log_lines?.length) {
            panelLines = panelLinesFromTexts(remote.log_lines);
          } else if (panelSteps.length > 0) {
            panelLines = stepsToPanelLogLines(panelSteps);
          }
          // enrich 专用：合并视图 steps + 各轮 steps（勿写回 panelSteps，否则 idle 会把历史步塞进「当前轮」）
          enrichSteps = mergeTraceStepsUnique(
            mergeTraceStepsUnique(panelSteps, parseTraceSteps(remote.steps)),
            panelTurns.flatMap((t) => t.steps),
          );
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
        let traceTurnsForEnrich = panelTurns;
        let traceStepsForEnrich = mergeTraceStepsUnique(enrichSteps, panelSteps);
        if (traceTurnsForEnrich.length === 0 && traceStepsForEnrich.length > 0) {
          traceTurnsForEnrich = parseTraceTurnsFromRemote(undefined, traceStepsForEnrich);
        }
        const chatMerged = dedupeUserMessages(
          dedupeConsecutiveUserMessages(
            finalizeAgentHistoryChat(parsed, {
              traceLines: panelLines.map((l) => l.text),
              traceTurns: traceTurnsForEnrich,
              traceSteps: traceStepsForEnrich,
              pendingText: pendingUser,
            }),
          ),
        );
        if (pendingUser && userMessageAlreadyInChat(chatMerged, pendingUser)) {
          clearPendingUserMessage(slug, node.thread_id);
        }
        setTraceLines([]);
        setTraceSteps([]);
        setStreamText('');
        setThinkingText('');
        traceLinesRef.current = [];
        traceStepsRef.current = [];
        if (sessionRunning) {
          const cached = getSessionChatCache(slug, node.thread_id);
          const finalMessages = hydrateChatMessageImages(
            mergeRunningSessionMessages(cached, chatMerged),
            slug,
            node.thread_id,
          );
          setMessages((prev) => preferRicherAgentChat(prev, finalMessages));
          setSessionChatCache(slug, node.thread_id, finalMessages);
        } else {
          const hydrated = hydrateChatMessageImages(chatMerged, slug, node.thread_id);
          setMessages((prev) => preferRicherAgentChat(prev, hydrated));
          setSessionChatCache(slug, node.thread_id, hydrated);
        }
        bumpContextRefresh();
        if (panelTurns.length === 0 && panelSteps.length > 0 && !sessionRunning) {
          // idle：整轮收成 completed turn。running 时保持 currentSteps，避免刷新后被 SSE 短回放盖成「第1轮·几步」
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
          if (panelSteps.length > 0) {
            setTraceSteps(panelSteps);
            traceStepsRef.current = panelSteps;
          }
        }
        setBusy(sessionRunning);
        ensureSessionSubscription(node.thread_id);
        void resumeSessionAfterSelect(node);
      }
      } finally {
        if (isCurrent()) {
          setHistoryLoading(false);
        }
      }
    },
    [slug, ensureSessionSubscription, resumeSessionAfterSelect, syncAgentRunningState, bumpContextRefresh],
  );

  return { loadHistory, historyLoading, loadSeqRef };
}
