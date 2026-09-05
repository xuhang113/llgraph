import type { Dispatch, MutableRefObject, SetStateAction } from 'react';
import { type TreeNode } from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import type { TraceStep } from '../types/trace';
import {
  appendTracePanelCacheLine,
  appendTracePanelCacheStep,
  saveTracePanelCache,
} from '../utils/tracePanelStore';
import {
  appendAssistantReplyIfMissing,
  extractMessageContent,
  formatAgentChatDisplayText,
  isUserVisibleAssistantText,
} from '../utils/messageText';
import {
  fetchEnrichedAgentChatMessages,
  preferRicherAgentChat,
} from '../utils/fetchEnrichedAgentChat';
import { clearPendingUserMessage } from '../utils/pendingUserMessage';
import type { TraceLine } from '../pages/console/types';
import {
  appendPanelTraceTurn,
  pushCompletedTraceTurn,
  appendUniquePanelLine,
  claimTurnStart,
  panelLinesFromTexts,
  parseTraceStep,
  parseTraceSteps,
  extractReplyTextFromTraceStep,
  releaseTurnOpen,
  traceLineSeenInCurrentTurn,
} from '../pages/console/traceUtils';
import { isExploreSubagentEvent } from '../pages/console/sseHelpers';

export type ConsoleSSEDeps = {
  slug: string;
  selected: TreeNode | null;
  selectedRef: MutableRefObject<TreeNode | null>;
  stoppedThreadsRef: MutableRefObject<Set<string>>;
  turnOpenRef: MutableRefObject<Set<string>>;
  runningThreadsRef: MutableRefObject<Set<string>>;
  streamAbortRef: MutableRefObject<Map<string, AbortController>>;
  streamLastEventAtRef: MutableRefObject<Map<string, number>>;
  panelTraceLinesRef: MutableRefObject<TraceLine[]>;
  panelTraceStepsRef: MutableRefObject<TraceStep[]>;
  panelTraceTurnsRef: MutableRefObject<import('../types/trace').TraceTurn[]>;
  traceLinesRef: MutableRefObject<TraceLine[]>;
  traceStepsRef: MutableRefObject<TraceStep[]>;
  traceTurnStartRef: MutableRefObject<number>;
  streamedRef: MutableRefObject<boolean>;
  traceFlushedRef: MutableRefObject<boolean>;
  setBusy: Dispatch<SetStateAction<boolean>>;
  setThinkingText: Dispatch<SetStateAction<string>>;
  setTraceActivitySec: Dispatch<SetStateAction<number>>;
  setCurrentTurnDurationSec: Dispatch<SetStateAction<number | null>>;
  setTraceLines: Dispatch<SetStateAction<TraceLine[]>>;
  setTraceSteps: Dispatch<SetStateAction<TraceStep[]>>;
  setPanelTraceLines: Dispatch<SetStateAction<TraceLine[]>>;
  setPanelTraceSteps: Dispatch<SetStateAction<TraceStep[]>>;
  setPanelTraceTurns: Dispatch<SetStateAction<import('../types/trace').TraceTurn[]>>;
  setStreamText: Dispatch<SetStateAction<string>>;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setFileChangesTick: Dispatch<SetStateAction<number>>;
  finalizeLiveTrace: (threadId?: string) => void;
  refreshTree: () => void;
  bumpContextRefresh: () => void;
};

export function createHandleSSEEvent(deps: ConsoleSSEDeps) {
  const {
    slug,
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
    traceTurnStartRef,
    streamedRef,
    setBusy,
    setThinkingText,
    setTraceActivitySec,
    setCurrentTurnDurationSec,
    setTraceLines,
    setTraceSteps,
    setPanelTraceLines,
    setPanelTraceSteps,
    setPanelTraceTurns,
    setStreamText,
    setMessages,
    setFileChangesTick,
    finalizeLiveTrace,
    refreshTree,
    bumpContextRefresh,
  } = deps;

  return (event: Record<string, unknown>, eventThread: string) => {
    const type = String(event.type || '');
    if (type === 'subagent_started' || type === 'subagent_updated') {
      refreshTree();
      return;
    }
    const traceWhileStopped =
      type === 'trace_line' ||
      type === 'trace_step' ||
      type === 'trace_activity' ||
      type === 'thinking_delta' ||
      type === 'stream_delta' ||
      type === 'stream_end';
    if (stoppedThreadsRef.current.has(eventThread) && traceWhileStopped) {
      return;
    }
    const sel = selectedRef.current;
    const agentView =
      (sel?.kind === 'agent' || sel?.kind === 'subagent') && eventThread === sel.thread_id;
    const viewing = agentView;
    const traceEventTypes = new Set([
      'trace_line',
      'trace_step',
      'trace_activity',
      'thinking_delta',
      'stream_delta',
      'stream_end',
      'turn_start',
      'turn_done',
    ]);
    const tracePanelThread = String(event.sub_thread || eventThread);

    if (agentView && isExploreSubagentEvent(event) && traceEventTypes.has(type)) {
      const sub = String(event.sub_thread || '').trim();
      if (slug && sub) {
        if (type === 'trace_line') {
          const line = String(event.text || '');
          if (line.trim()) {
            appendTracePanelCacheLine(slug, sub, line);
          }
        } else if (type === 'trace_step') {
          const raw = event.step as Record<string, unknown> | undefined;
          if (raw) {
            appendTracePanelCacheStep(slug, sub, parseTraceStep(raw));
          }
        }
      }
      if (type !== 'end' && type !== 'error' && type !== 'interrupt') {
        return;
      }
    }

    if (type === 'turn_start') {
      runningThreadsRef.current.add(eventThread);
      if (agentView) {
        setBusy(true);
      }
    } else if (type === 'end') {
      runningThreadsRef.current.delete(eventThread);
      streamAbortRef.current.delete(eventThread);
      streamLastEventAtRef.current.delete(eventThread);
      releaseTurnOpen(eventThread, turnOpenRef.current);
      stoppedThreadsRef.current.delete(eventThread);
      const selNow = selectedRef.current;
      if (
        (selNow?.kind === 'agent' || selNow?.kind === 'subagent') &&
        selNow.thread_id === eventThread
      ) {
        setBusy(false);
      }
      if (slug && eventThread.startsWith('cli-')) {
        clearPendingUserMessage(slug, eventThread);
      }
    }

    if (!viewing) {
      if (!slug) {
        return;
      }
      if (type === 'turn_start') {
        if (claimTurnStart(eventThread, turnOpenRef.current)) {
          appendTracePanelCacheLine(
            slug,
            eventThread,
            `─── 本轮 ${new Date().toLocaleTimeString('zh-CN', { hour12: false })} ───`,
          );
        }
      } else if (type === 'trace_line') {
        const line = String(event.text || '');
        if (line.trim() && !traceLineSeenInCurrentTurn(panelTraceLinesRef.current, line)) {
          appendTracePanelCacheLine(slug, eventThread, line);
        }
      } else if (type === 'trace_step') {
        const raw = event.step as Record<string, unknown> | undefined;
        if (raw) {
          appendTracePanelCacheStep(slug, eventThread, parseTraceStep(raw));
        }
      } else if (type === 'end') {
        refreshTree();
      }
      return;
    }

    if (type === 'turn_start') {
      setThinkingText('');
      traceTurnStartRef.current = Date.now();
      setTraceActivitySec(0);
      setCurrentTurnDurationSec(null);
      if (event.replay === true) {
        turnOpenRef.current.add(eventThread);
      } else if (claimTurnStart(eventThread, turnOpenRef.current)) {
        const turnNum = panelTraceTurnsRef.current.length + 1;
        pushCompletedTraceTurn(
          panelTraceTurnsRef,
          panelTraceStepsRef,
          setPanelTraceTurns,
          setPanelTraceSteps,
          `第 ${turnNum} 轮`,
        );
        setTraceLines([]);
        setTraceSteps([]);
        traceLinesRef.current = [];
        traceStepsRef.current = [];
        const entry = {
          id: `turn-sep-${Date.now()}`,
          text: `─── 本轮 ${new Date().toLocaleTimeString('zh-CN', { hour12: false })} ───`,
        };
        const nextLines = [...panelTraceLinesRef.current, entry];
        panelTraceLinesRef.current = nextLines;
        setPanelTraceLines(nextLines);
      }
    } else if (type === 'trace_line') {
      const line = String(event.text || '');
      if (line.trim()) {
        if (
          event.replay === true &&
          traceLineSeenInCurrentTurn(panelTraceLinesRef.current, line)
        ) {
          return;
        }
        if (traceLineSeenInCurrentTurn(panelTraceLinesRef.current, line)) {
          return;
        }
        setPanelTraceLines((prev) => {
          const next = appendUniquePanelLine(prev, line);
          panelTraceLinesRef.current = next;
          return next;
        });
        setTraceLines((prev) => {
          const next = appendUniquePanelLine(prev, line);
          traceLinesRef.current = next;
          return next;
        });
        if (slug && agentView) {
          appendTracePanelCacheLine(slug, eventThread, line);
        }
      }
    } else if (type === 'trace_activity') {
      const sec = Number(event.elapsed_sec);
      if (Number.isFinite(sec) && sec >= 0) {
        setTraceActivitySec(Math.floor(sec));
      }
    } else if (type === 'thinking_delta') {
      if (event.replay === true) {
        return;
      }
      const text = String(event.text || '');
      if (text.trim()) {
        setThinkingText(text);
      }
    } else if (type === 'stream_delta') {
      if (event.replay === true) {
        return;
      }
      streamedRef.current = true;
      setStreamText((s) => s + String(event.text || ''));
    } else if (type === 'stream_end') {
      if (!streamedRef.current) {
        setStreamText('');
      }
    } else if (type === 'trace_step') {
      const raw = event.step as Record<string, unknown> | undefined;
      if (raw) {
        const step = parseTraceStep(raw);
        const exists = panelTraceStepsRef.current.some((s) => s.step_id === step.step_id);
        if (event.replay === true && exists) {
          return;
        }
        setPanelTraceSteps((prev) => {
          const idx = prev.findIndex((s) => s.step_id === step.step_id);
          const next =
            idx >= 0
              ? prev.map((s, i) => (i === idx ? step : s))
              : [...prev, step];
          panelTraceStepsRef.current = next;
          return next;
        });
        const liveIdx = traceStepsRef.current.findIndex((s) => s.step_id === step.step_id);
        if (liveIdx >= 0) {
          const next = traceStepsRef.current.map((s, i) => (i === liveIdx ? step : s));
          traceStepsRef.current = next;
          setTraceSteps(next);
        } else {
          const next = [...traceStepsRef.current, step];
          traceStepsRef.current = next;
          setTraceSteps(next);
        }
        if (slug && tracePanelThread) {
          appendTracePanelCacheStep(slug, tracePanelThread, step);
        }
        if (agentView && step.kind === 'reply') {
          const replyPreview = formatAgentChatDisplayText(extractReplyTextFromTraceStep(step));
          if (isUserVisibleAssistantText(replyPreview)) {
            setMessages((prev) => appendAssistantReplyIfMissing(prev, replyPreview));
            streamedRef.current = true;
            setStreamText(replyPreview);
          }
        }
      }
    } else if (type === 'turn_done') {
      if (event.replay === true) {
        return;
      }
      const fallbackSteps = parseTraceSteps(event.trace_steps);
      if (fallbackSteps.length > 0) {
        appendPanelTraceTurn(
          panelTraceLinesRef,
          panelTraceStepsRef,
          fallbackSteps,
          setPanelTraceLines,
          setPanelTraceSteps,
        );
      } else if (traceLinesRef.current.length > 0 || panelTraceLinesRef.current.length > 0) {
        const liveLines = panelLinesFromTexts(traceLinesRef.current.map((l) => l.text));
        if (liveLines.length > 0) {
          const nextLines = [...panelTraceLinesRef.current, ...liveLines];
          panelTraceLinesRef.current = nextLines;
          setPanelTraceLines(nextLines);
        }
      }
      if (slug && tracePanelThread) {
        saveTracePanelCache(
          slug,
          tracePanelThread,
          panelTraceLinesRef.current.map((line) => line.text),
          panelTraceStepsRef.current,
        );
      }
      finalizeLiveTrace(tracePanelThread);
      releaseTurnOpen(eventThread, turnOpenRef.current);
      const durationSec = Number(event.duration_sec);
      if (Number.isFinite(durationSec) && durationSec > 0) {
        setCurrentTurnDurationSec(durationSec);
      }
      runningThreadsRef.current.delete(eventThread);
      streamAbortRef.current.delete(eventThread);
      streamLastEventAtRef.current.delete(eventThread);
      const stepPool = [
        ...panelTraceStepsRef.current,
        ...parseTraceSteps(event.trace_steps),
      ];
      const replyFromSteps = stepPool
        .filter((s) => s.kind === 'reply')
        .map((s) => formatAgentChatDisplayText(extractReplyTextFromTraceStep(s)))
        .filter((t) => isUserVisibleAssistantText(t));
      const fallbackReply = formatAgentChatDisplayText(extractMessageContent(event.text));
      const effectiveReply = isUserVisibleAssistantText(fallbackReply)
        ? fallbackReply
        : (replyFromSteps[replyFromSteps.length - 1] || '');
      setBusy(false);
      setThinkingText('');
      setTraceActivitySec(0);
      if (isUserVisibleAssistantText(effectiveReply)) {
        setMessages((prev) => appendAssistantReplyIfMissing(prev, effectiveReply));
      }
      setStreamText('');
      if (slug && agentView && sel?.thread_id === eventThread) {
        void fetchEnrichedAgentChatMessages(slug, eventThread, {
          fallbackReply: effectiveReply,
          liveLines: panelTraceLinesRef.current,
          liveSteps: stepPool,
          liveTurns: panelTraceTurnsRef.current,
        })
          .then((parsed) => {
            if (selectedRef.current?.thread_id !== eventThread) {
              return;
            }
            setMessages((prev) => preferRicherAgentChat(prev, parsed));
          })
          .catch(() => {});
      }
      streamedRef.current = false;
      setFileChangesTick((n) => n + 1);
      bumpContextRefresh();
    } else if (type === 'title_updated') {
      refreshTree();
    } else if (type === 'interrupt') {
      const payload = event.payload as Record<string, unknown>;
      if (payload?.type === 'user_stop') {
        stoppedThreadsRef.current.delete(eventThread);
        runningThreadsRef.current.delete(eventThread);
        streamAbortRef.current.delete(eventThread);
        streamLastEventAtRef.current.delete(eventThread);
        releaseTurnOpen(eventThread, turnOpenRef.current);
        finalizeLiveTrace(eventThread);
        if (agentView && sel?.thread_id === eventThread) {
          setBusy(false);
          setStreamText('');
          setThinkingText('');
          setTraceActivitySec(0);
          setMessages((prev) => [
            ...prev,
            {
              id: `stop-done-${Date.now()}`,
              role: 'system',
              text: String(payload.message || '已停止当前生成。'),
            },
          ]);
        }
      }
    } else if (type === 'error') {
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'system', text: `**错误:** ${event.message}` },
      ]);
    } else if (type === 'end') {
      finalizeLiveTrace(tracePanelThread);
      runningThreadsRef.current.delete(eventThread);
      streamAbortRef.current.delete(eventThread);
      streamLastEventAtRef.current.delete(eventThread);
      if (agentView && sel?.thread_id === eventThread) {
        setBusy(false);
      }
      setTraceActivitySec(0);
      traceTurnStartRef.current = 0;
      streamedRef.current = false;
      refreshTree();
      bumpContextRefresh();
    }
  };
}
