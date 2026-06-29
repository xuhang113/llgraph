import type { Dispatch, MutableRefObject, SetStateAction } from 'react';
import {
  api,
  type PlanDetail,
  type SurveySpec,
  type TreeNode,
} from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import type { TraceStep } from '../types/trace';
import {
  appendTracePanelCacheLine,
  appendTracePanelCacheStep,
  saveTracePanelCache,
} from '../utils/tracePanelStore';
import {
  dedupeConsecutiveUserMessages,
  extractMessageContent,
  formatAgentChatDisplayText,
  parseAgentHistoryMessages,
} from '../utils/messageText';
import { clearPendingUserMessage } from '../utils/pendingUserMessage';
import { buildPlanChatMessages } from '../utils/planChat';
import type { TraceLine } from '../pages/console/types';
import { planExecutionAllowWrite } from '../pages/console/planHelpers';
import {
  appendPanelTraceTurn,
  pushCompletedTraceTurn,
  appendUniquePanelLine,
  claimTurnStart,
  panelLinesFromTexts,
  parseTraceStep,
  parseTraceSteps,
  releaseTurnOpen,
  traceLineSeenInCurrentTurn,
} from '../pages/console/traceUtils';
import { parseWorkerMessages } from '../pages/console/workerUtils';
import {
  eventMatchesWorkerTask,
  isPlannerSubagentEvent,
  isSubagentWorkerEvent,
} from '../pages/console/sseHelpers';
import {
  applyPendingConfirmHead,
  ingestPlanConfirmPayload,
  ingestSurveyConfirm,
  ingestTaskStepConfirm,
} from '../utils/pendingConfirmUi';

function resolvePlanThreadForEvent(
  sel: TreeNode | null,
  eventThread: string,
  workerView: boolean,
  workerPlanThread: string,
): string {
  if (sel?.kind === 'plan') {
    return sel.thread_id;
  }
  if (workerView && workerPlanThread) {
    return workerPlanThread;
  }
  if (
    eventThread.startsWith('plan-')
    && !eventThread.includes(':planner:')
    && !eventThread.includes(':worker:')
  ) {
    return eventThread;
  }
  return eventThread;
}

function selectionMatchesConfirmThread(
  sel: TreeNode | null,
  confirmThread: string,
  eventThread: string,
  workerPlanThread: string,
): boolean {
  if (!sel) {
    return false;
  }
  if (sel.kind === 'agent') {
    return sel.thread_id === confirmThread || sel.thread_id === eventThread;
  }
  if (sel.kind === 'plan') {
    return sel.thread_id === confirmThread;
  }
  if (sel.kind === 'worker') {
    return workerPlanThread === confirmThread;
  }
  return false;
}

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
  confirmAutoShownRef: MutableRefObject<string | null>;
  surveyDismissedRef: MutableRefObject<string | null>;
  taskStepDismissedRef: MutableRefObject<string | null>;
  traceTurnStartRef: MutableRefObject<number>;
  streamedRef: MutableRefObject<boolean>;
  traceFlushedRef: MutableRefObject<boolean>;
  setBusy: Dispatch<SetStateAction<boolean>>;
  setThinkingText: Dispatch<SetStateAction<string>>;
  setTraceActivitySec: Dispatch<SetStateAction<number>>;
  setTraceLines: Dispatch<SetStateAction<TraceLine[]>>;
  setTraceSteps: Dispatch<SetStateAction<TraceStep[]>>;
  setPanelTraceLines: Dispatch<SetStateAction<TraceLine[]>>;
  setPanelTraceSteps: Dispatch<SetStateAction<TraceStep[]>>;
  setPanelTraceTurns: Dispatch<SetStateAction<import('../types/trace').TraceTurn[]>>;
  setStreamText: Dispatch<SetStateAction<string>>;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setPlanDetail: Dispatch<SetStateAction<PlanDetail | null>>;
  setAllowWrite: Dispatch<SetStateAction<boolean>>;
  setSurvey: Dispatch<SetStateAction<SurveySpec | null>>;
  setPlanConfirm: Dispatch<SetStateAction<Record<string, unknown> | null>>;
  setTaskStepConfirm: Dispatch<SetStateAction<string | null>>;
  setFileChangesTick: Dispatch<SetStateAction<number>>;
  ensurePlanSubscription: (planThread: string) => void;
  maybeReleasePlanSubscription: (planThread: string) => void;
  finalizeLiveTrace: (threadId?: string) => void;
  refreshTree: () => void;
  bumpContextRefresh: () => void;
};

export function createHandleSSEEvent(deps: ConsoleSSEDeps) {
  const {
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
  } = deps;

  return (event: Record<string, unknown>, eventThread: string) => {
    const type = String(event.type || '');
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
    const planMainThread = sel?.kind === 'plan' ? sel.thread_id : '';
    const isPlanPlannerThread =
      Boolean(planMainThread) &&
      (eventThread.startsWith(`${planMainThread}:planner:`) ||
        (isPlannerSubagentEvent(event) &&
          String(event.sub_thread || eventThread).startsWith(`${planMainThread}:planner:`)));
    const planView =
      sel?.kind === 'plan' && (sel.thread_id === eventThread || isPlanPlannerThread);
    const workerPlanThread =
      sel?.kind === 'worker' ? sel.thread_id.split(':worker:')[0] : '';
    const workerTaskId =
      sel?.kind === 'worker'
        ? sel.task_id || sel.thread_id.split(':worker:')[1] || ''
        : '';
    const workerView =
      sel?.kind === 'worker' &&
      (eventThread === sel.thread_id || eventThread === workerPlanThread);
    const agentView = sel?.kind === 'agent' && eventThread === sel.thread_id;
    const viewing = planView || workerView || agentView;
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
    const isWorkerTrace =
      isSubagentWorkerEvent(event) && traceEventTypes.has(type);
    const tracePanelThread = workerView
      ? sel!.thread_id
      : String(event.sub_thread || eventThread);

    if (planView && isWorkerTrace) {
      if (slug && event.sub_thread) {
        const sub = String(event.sub_thread);
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
      if (type !== 'plan_state' && type !== 'plan_job' && type !== 'end' && type !== 'error' && type !== 'interrupt') {
        return;
      }
    }

    if (
      planView &&
      (isPlannerSubagentEvent(event) || isPlanPlannerThread) &&
      traceEventTypes.has(type)
    ) {
      const sub = String(event.sub_thread || eventThread);
      if (slug && sub) {
        if (type === 'trace_line') {
          const line = String(event.text || '');
          if (line.trim()) {
            appendTracePanelCacheLine(slug, sub, line);
            const entry = { id: `p-${Date.now()}-${Math.random()}`, text: line };
            setPanelTraceLines((prev) => {
              const next = [...prev, entry];
              panelTraceLinesRef.current = next;
              return next;
            });
          }
        } else if (type === 'trace_step') {
          const raw = event.step as Record<string, unknown> | undefined;
          if (raw) {
            const step = parseTraceStep(raw);
            appendTracePanelCacheStep(slug, sub, step);
            setPanelTraceSteps((prev) => {
              const next = [...prev, step];
              panelTraceStepsRef.current = next;
              return next;
            });
          }
        } else if (type === 'turn_start') {
          if (claimTurnStart(sub, turnOpenRef.current)) {
            const entry = {
              id: `turn-sep-${Date.now()}`,
              text: `─── 本轮 ${new Date().toLocaleTimeString('zh-CN', { hour12: false })} ───`,
            };
            appendTracePanelCacheLine(slug, sub, entry.text);
            setPanelTraceLines((prev) => {
              const next = [...prev, entry];
              panelTraceLinesRef.current = next;
              return next;
            });
          }
        }
      }
      if (type !== 'plan_state' && type !== 'plan_job' && type !== 'end' && type !== 'error' && type !== 'interrupt') {
        return;
      }
    }

    if (workerView && isWorkerTrace && !eventMatchesWorkerTask(event, sel!.thread_id, workerTaskId)) {
      return;
    }

    if (type === 'turn_start') {
      runningThreadsRef.current.add(eventThread);
      if (agentView) {
        setBusy(true);
      }
      if (
        eventThread.startsWith('plan-') &&
        !eventThread.includes(':planner:') &&
        !eventThread.includes(':worker:')
      ) {
        ensurePlanSubscription(eventThread);
      }
    } else if (type === 'end') {
      const isPlanMain =
        eventThread.startsWith('plan-') &&
        !eventThread.includes(':planner:') &&
        !eventThread.includes(':worker:');
      if (isPlanMain && slug) {
        void api
          .plan(slug, eventThread)
          .then((detail) => {
            if (!detail.job?.running) {
              runningThreadsRef.current.delete(eventThread);
              streamAbortRef.current.delete(eventThread);
            } else {
              runningThreadsRef.current.add(eventThread);
            }
            if (
              selectedRef.current?.kind === 'plan' &&
              selectedRef.current.thread_id === eventThread
            ) {
              setPlanDetail(detail);
              setBusy(Boolean(detail.job?.running));
            }
          })
          .catch(() => {
            runningThreadsRef.current.delete(eventThread);
            streamAbortRef.current.delete(eventThread);
          });
      } else {
        runningThreadsRef.current.delete(eventThread);
        streamAbortRef.current.delete(eventThread);
        streamLastEventAtRef.current.delete(eventThread);
        releaseTurnOpen(eventThread, turnOpenRef.current);
        stoppedThreadsRef.current.delete(eventThread);
        const selNow = selectedRef.current;
        if (selNow?.kind === 'agent' && selNow.thread_id === eventThread) {
          setBusy(false);
        }
        if (slug && eventThread.startsWith('cli-')) {
          clearPendingUserMessage(slug, eventThread);
        }
      }
      maybeReleasePlanSubscription(eventThread);
    }

    const pendingUiSetters = { setSurvey, setPlanConfirm, setTaskStepConfirm };
    const maybeApplyPending = (confirmThread: string) => {
      if (
        slug
        && selectionMatchesConfirmThread(sel, confirmThread, eventThread, workerPlanThread)
      ) {
        applyPendingConfirmHead(slug, confirmThread, pendingUiSetters, {
          surveyDismissedId: surveyDismissedRef.current,
          taskStepDismissedId: taskStepDismissedRef.current,
        });
      }
    };

    if (slug) {
      if ((type === 'survey' || type === 'turn_done') && event.survey) {
        if (event.replay !== true) {
          ingestSurveyConfirm(slug, eventThread, event.survey as SurveySpec);
        }
        maybeApplyPending(eventThread);
      }
      if (type === 'interrupt') {
        const payload = event.payload as Record<string, unknown> | undefined;
        if (payload?.type === 'plan_confirm') {
          const planThread = resolvePlanThreadForEvent(
            sel,
            eventThread,
            workerView,
            workerPlanThread,
          );
          ingestPlanConfirmPayload(slug, planThread, payload);
          confirmAutoShownRef.current = null;
          maybeApplyPending(planThread);
        } else if (payload?.type === 'task_step_confirm') {
          const planThread = resolvePlanThreadForEvent(
            sel,
            eventThread,
            workerView,
            workerPlanThread,
          );
          const taskId = String(payload.task_id || '');
          if (taskId) {
            ingestTaskStepConfirm(slug, planThread, taskId);
            taskStepDismissedRef.current = null;
            maybeApplyPending(planThread);
          }
        }
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
      } else if (type === 'turn_done' || type === 'survey') {
        /* steps 已由 trace_step 增量写入；turn_done 不再重复合并 */
      } else if (type === 'end') {
        refreshTree();
      }
      return;
    }

    if (type === 'turn_start') {
      setThinkingText('');
      traceTurnStartRef.current = Date.now();
      setTraceActivitySec(0);
      if (claimTurnStart(eventThread, turnOpenRef.current)) {
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
      setStreamText('');
    } else if (type === 'trace_step') {
      const raw = event.step as Record<string, unknown> | undefined;
      if (raw) {
        const step = parseTraceStep(raw);
        const exists = panelTraceStepsRef.current.some((s) => s.step_id === step.step_id);
        if (event.replay === true && exists) {
          return;
        }
        if (!exists) {
          setPanelTraceSteps((prev) => {
            const next = [...prev, step];
            panelTraceStepsRef.current = next;
            return next;
          });
        }
        const liveExists = traceStepsRef.current.some((s) => s.step_id === step.step_id);
        if (!liveExists) {
          const next = [...traceStepsRef.current, step];
          traceStepsRef.current = next;
          setTraceSteps(next);
        }
        if (slug && tracePanelThread) {
          appendTracePanelCacheStep(slug, tracePanelThread, step);
        }
      }
    } else if (type === 'turn_done' || type === 'survey') {
      const isReplay = event.replay === true;
      if (isReplay) {
        maybeApplyPending(eventThread);
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
      const fallbackReply = formatAgentChatDisplayText(extractMessageContent(event.text));
      if (agentView) {
        setBusy(false);
        setStreamText('');
        setThinkingText('');
        setTraceActivitySec(0);
        if (slug && sel?.kind === 'agent' && sel.thread_id === eventThread) {
          void api
            .messages(slug, eventThread)
            .then((data) => {
              const parsed = dedupeConsecutiveUserMessages(
                parseAgentHistoryMessages(data.messages || []),
              );
              setMessages(parsed);
            })
            .catch(() => {
              if (fallbackReply.trim()) {
                setMessages((prev) => [
                  ...prev,
                  { id: `done-${Date.now()}`, role: 'assistant', text: fallbackReply },
                ]);
              }
            });
        } else if (fallbackReply.trim()) {
          setMessages((prev) => [
            ...prev,
            { id: `done-${Date.now()}`, role: 'assistant', text: fallbackReply },
          ]);
        }
      } else {
        if (fallbackReply.trim()) {
          setMessages((prev) => [
            ...prev,
            { id: `done-${Date.now()}`, role: 'assistant', text: fallbackReply },
          ]);
        }
      }
      streamedRef.current = false;
      maybeApplyPending(eventThread);
      if (selected?.kind === 'agent' || selected?.kind === 'plan' || selected?.kind === 'worker') {
        setFileChangesTick((n) => n + 1);
      }
      bumpContextRefresh();
    } else if (type === 'interrupt') {
      const payload = event.payload as Record<string, unknown>;
      if (payload?.type === 'tasks_incomplete') {
        setMessages((prev) => [
          ...prev,
          {
            id: `intr-${Date.now()}`,
            role: 'system',
            text: String(payload.message || '仍有未完成任务，可在 Plan 面板继续执行'),
          },
        ]);
      } else if (payload?.type === 'user_stop') {
        stoppedThreadsRef.current.delete(eventThread);
        runningThreadsRef.current.delete(eventThread);
        streamAbortRef.current.delete(eventThread);
        streamLastEventAtRef.current.delete(eventThread);
        releaseTurnOpen(eventThread, turnOpenRef.current);
        finalizeLiveTrace(
          agentView ? eventThread : tracePanelThread,
        );
        if (sel?.kind === 'agent' && sel.thread_id === eventThread) {
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
    } else if (type === 'plan_job') {
      const running = Boolean(event.running);
      const jobError = event.error != null ? String(event.error) : null;
      const planThreadForJob =
        planView && sel?.kind === 'plan'
          ? sel.thread_id
          : workerView
            ? workerPlanThread
            : eventThread.startsWith('plan-') &&
                !eventThread.includes(':planner:') &&
                !eventThread.includes(':worker:')
              ? eventThread
              : '';
      if (running && planThreadForJob) {
        runningThreadsRef.current.add(planThreadForJob);
        setBusy(true);
      } else if (!running && planThreadForJob) {
        runningThreadsRef.current.delete(planThreadForJob);
      }
      setPlanDetail((prev) =>
        prev ? { ...prev, job: { running, error: jobError } } : prev,
      );
      if (planThreadForJob && slug) {
        void api
          .plan(slug, planThreadForJob)
          .then((detail) => {
            setPlanDetail(detail);
            if (planExecutionAllowWrite(detail)) {
              setAllowWrite(true);
            }
            if (!detail.job?.running) {
              setBusy(false);
            }
          })
          .catch(() => {});
      }
      if (workerView && slug && workerTaskId) {
        void Promise.all([
          api.worker(slug, workerPlanThread, workerTaskId),
          api.plan(slug, workerPlanThread),
        ]).then(([data, detail]) => {
          setPlanDetail(detail);
          const parsed = parseWorkerMessages(data);
          setMessages(parsed.chat);
          refreshTree();
        }).catch(() => {});
      }
    } else if (type === 'plan_done') {
      setFileChangesTick((n) => n + 1);
      if (slug && selected?.kind === 'plan' && eventThread === selected.thread_id) {
        api.plan(slug, selected.thread_id).then(async (detail) => {
          setPlanDetail(detail);
          try {
            const data = await api.messages(slug, selected.thread_id);
            setMessages(buildPlanChatMessages(detail, data.messages || []));
          } catch {
            setMessages(buildPlanChatMessages(detail));
          }
        }).catch(() => {});
      }
      refreshTree();
      bumpContextRefresh();
    } else if (type === 'plan_state') {
      if (sel?.kind === 'plan' && slug) {
        api.plan(slug, sel.thread_id).then(setPlanDetail);
      }
      if (sel?.kind === 'worker' && slug) {
        const pt = sel.thread_id.split(':worker:')[0];
        api.plan(slug, pt).then(setPlanDetail);
      }
    } else if (type === 'error') {
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'system', text: `**错误:** ${event.message}` },
      ]);
      if (sel?.kind === 'plan' && slug) {
        api.plan(slug, sel.thread_id).then(setPlanDetail).catch(() => {});
      }
    } else if (type === 'end') {
      finalizeLiveTrace(tracePanelThread);
      setBusy(false);
      setTraceActivitySec(0);
      traceTurnStartRef.current = 0;
      streamedRef.current = false;
      refreshTree();
      bumpContextRefresh();
    }
  }
}
