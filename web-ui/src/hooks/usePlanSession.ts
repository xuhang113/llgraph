import { useCallback, useEffect } from 'react';
import { api, type PlanDetail, type TreeNode } from '../api/client';
import type { ChatMessage } from '../components/console/ChatThread';
import { planExecutionAllowWrite, planNeedsConfirm } from '../pages/console/planHelpers';
import { findPlanNode } from '../pages/console/workerUtils';
import {
  applyPendingConfirmHead,
  ingestPlanConfirmFromDetail,
  ingestTaskStepConfirm,
} from '../utils/pendingConfirmUi';
import { clearConfirmQueue, peekConfirmHead } from '../utils/pendingConfirmQueue';
import { releaseStreamState } from '../pages/console/streamHelpers';

export type PlanSessionDeps = {
  slug: string;
  selected: TreeNode | null;
  planDetail: PlanDetail | null;
  allowWrite: boolean;
  plans: TreeNode[];
  alert: (message: string) => Promise<void>;
  confirm: (message: string) => Promise<boolean>;
  setBusy: React.Dispatch<React.SetStateAction<boolean>>;
  setPlanDetail: React.Dispatch<React.SetStateAction<PlanDetail | null>>;
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  setAllowWrite: React.Dispatch<React.SetStateAction<boolean>>;
  setPlanConfirm: React.Dispatch<React.SetStateAction<Record<string, unknown> | null>>;
  setTaskStepConfirm: React.Dispatch<React.SetStateAction<string | null>>;
  taskStepDismissedRef: React.MutableRefObject<string | null>;
  planStopInFlightRef: React.MutableRefObject<boolean>;
  runningThreadsRef: React.MutableRefObject<Set<string>>;
  streamAbortRef: React.MutableRefObject<Map<string, AbortController>>;
  streamLastEventAtRef: React.MutableRefObject<Map<string, number>>;
  beginStream: (threadId: string) => AbortController;
  bindSSE: (threadId: string) => (ev: Record<string, unknown>) => void;
  finalizeLiveTrace: (threadId?: string) => void;
  handleSelect: (node: TreeNode) => void;
  refreshTree: () => void;
};

export function usePlanSession(deps: PlanSessionDeps) {
  const {
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
    planStopInFlightRef,
    runningThreadsRef,
    streamAbortRef,
    streamLastEventAtRef,
    beginStream,
    bindSSE,
    finalizeLiveTrace,
    handleSelect,
    refreshTree,
  } = deps;

  const streamRefs = {
    runningThreads: runningThreadsRef,
    streamAbort: streamAbortRef,
    streamLastEventAt: streamLastEventAtRef,
  };

  const releasePlanStream = (threadId: string, clearBusy = true) => {
    releaseStreamState(threadId, streamRefs, clearBusy ? setBusy : undefined);
  };

  const maybePromptPlanConfirm = useCallback((detail: PlanDetail, threadId: string) => {
    if (!planNeedsConfirm(detail) || !slug) {
      return;
    }
    ingestPlanConfirmFromDetail(slug, threadId, detail);
    applyPendingConfirmHead(slug, threadId, {
      setSurvey: () => undefined,
      setPlanConfirm,
      setTaskStepConfirm,
    });
  }, [slug, setPlanConfirm, setTaskStepConfirm]);

  useEffect(() => {
    if (!slug || selected?.kind !== 'plan' || !planDetail) {
      return;
    }
    if (planNeedsConfirm(planDetail)) {
      ingestPlanConfirmFromDetail(slug, selected.thread_id, planDetail);
    } else {
      clearConfirmQueue(slug, selected.thread_id, 'plan_confirm');
      setPlanConfirm(null);
    }
  }, [selected, planDetail, slug, setPlanConfirm]);
  const handleWorkerSelect = (taskId: string) => {
    if (!selected || selected.kind !== 'plan') {
      return;
    }
    const task = planDetail?.tasks.find((t) => String(t.id) === taskId);
    const wfTask = planDetail?.workflow_snapshot?.tasks?.find((t) => t.id === taskId);
    const workerNode: TreeNode = {
      kind: 'worker',
      thread_id: `${selected.thread_id}:worker:${taskId}`,
      task_id: taskId,
      title: String(task?.title || taskId),
      status: String(wfTask?.status || task?.status || 'pending'),
      children: [],
    };
    handleSelect(workerNode);
  };

  const handleWorkClick = async (taskId: string) => {
    if (!slug || !selected || selected.kind !== 'plan' || !planDetail) {
      return;
    }
    const task = planDetail.tasks.find((t) => String(t.id) === taskId);
    const wfTask = planDetail.workflow_snapshot?.tasks?.find((t) => t.id === taskId);
    const status = String(wfTask?.worker_node_status || wfTask?.status || task?.status || 'pending');
    const phase = planDetail.phase;

    if (phase === 'completed') {
      handleWorkerSelect(taskId);
      return;
    }

    if (phase === 'awaiting_confirm' || phase === 'planning') {
      if (phase === 'awaiting_confirm') {
        await alert('请先确认计划后再执行 Work');
      } else {
        handleWorkerSelect(taskId);
      }
      return;
    }

    if (status === 'failed' && phase === 'executing') {
      try {
        const check = await api.planTaskRunnable(slug, selected.thread_id, taskId);
        if (check.ok) {
          setBusy(true);
          const ac = beginStream(selected.thread_id);
          try {
            await api.planRunTask(
              slug,
              selected.thread_id,
              taskId,
              allowWrite,
              bindSSE(selected.thread_id),
              ac.signal,
            );
            return;
          } catch (err) {
            releasePlanStream(selected.thread_id);
            await alert(err instanceof Error ? err.message : String(err));
            return;
          }
        }
        await alert(check.message || '当前 Work 不可执行');
        return;
      } catch (err) {
        releasePlanStream(selected.thread_id);
        await alert(err instanceof Error ? err.message : String(err));
        return;
      } finally {
        if (!runningThreadsRef.current.has(selected.thread_id)) {
          setBusy(false);
        }
      }
    }

    if (status === 'done') {
      if (!task?.readonly && phase === 'executing') {
        try {
          const check = await api.planTaskRunnable(slug, selected.thread_id, taskId);
          if (check.ok) {
            setBusy(true);
            const ac = beginStream(selected.thread_id);
            try {
              await api.planRunTask(
                slug,
                selected.thread_id,
                taskId,
                allowWrite,
                bindSSE(selected.thread_id),
                ac.signal,
              );
              return;
            } catch (err) {
              releasePlanStream(selected.thread_id);
              /* 不可重跑则查看详情 */
            }
          }
        } catch {
          /* 不可重跑则查看详情 */
        } finally {
          if (!runningThreadsRef.current.has(selected.thread_id)) {
            setBusy(false);
          }
        }
      }
      handleWorkerSelect(taskId);
      return;
    }

    if (status === 'running' && planDetail.job?.running) {
      handleWorkerSelect(taskId);
      return;
    }

    if (phase !== 'executing') {
      handleWorkerSelect(taskId);
      return;
    }

    try {
      const check = await api.planTaskRunnable(slug, selected.thread_id, taskId);
      if (!check.ok) {
        await alert(check.message || '当前 Work 不可执行');
        return;
      }
      setBusy(true);
      const ac = beginStream(selected.thread_id);
      await api.planRunTask(
        slug,
        selected.thread_id,
        taskId,
        allowWrite,
        bindSSE(selected.thread_id),
        ac.signal,
      );
    } catch (err) {
      releasePlanStream(selected.thread_id);
      await alert(err instanceof Error ? err.message : String(err));
    } finally {
      if (!runningThreadsRef.current.has(selected.thread_id)) {
        setBusy(false);
      }
    }
  };

  const handleBackToPlan = () => {
    if (!selected || selected.kind !== 'worker') {
      return;
    }
    const planThread = selected.thread_id.split(':worker:')[0];
    const planNode = findPlanNode(plans, planThread);
    if (planNode) {
      handleSelect(planNode);
      return;
    }
    handleSelect({
      kind: 'plan',
      thread_id: planThread,
      title: planDetail?.title || planThread,
      children: [],
    });
  };
  const handlePlanStop = async () => {
    if (!slug || !selected || selected.kind !== 'plan' || planStopInFlightRef.current) {
      return;
    }
    planStopInFlightRef.current = true;
    const tid = selected.thread_id;
    streamAbortRef.current.get(tid)?.abort();
    releasePlanStream(tid);
    finalizeLiveTrace(tid);
    clearConfirmQueue(slug, tid, 'plan_confirm');
    clearConfirmQueue(slug, tid, 'task_step_confirm');
    setPlanConfirm(null);
    setTaskStepConfirm(null);
    try {
      const res = await api.planCancel(slug, tid);
      if (!res.ok) {
        await alert(res.message);
      }
      const detail = await api.plan(slug, tid);
      setPlanDetail(detail);
      refreshTree();
    } catch (err) {
      await alert(err instanceof Error ? err.message : String(err));
    } finally {
      planStopInFlightRef.current = false;
    }
  };

  const handlePlanAbort = async () => {
    if (!slug || !selected || selected.kind !== 'plan') {
      return;
    }
    const ok = await confirm(
      '取消 Plan 后，未完成的 Work 将标记为跳过，且不会继续汇总。确定取消？',
    );
    if (!ok) {
      return;
    }
    const tid = selected.thread_id;
    streamAbortRef.current.get(tid)?.abort();
    releasePlanStream(tid);
    clearConfirmQueue(slug, tid, 'plan_confirm');
    clearConfirmQueue(slug, tid, 'task_step_confirm');
    setPlanConfirm(null);
    setTaskStepConfirm(null);
    try {
      const res = await api.planAbort(slug, tid);
      if (!res.ok) {
        await alert(res.message);
      } else {
        setMessages((prev) => [
          ...prev,
          { id: `abort-${Date.now()}`, role: 'system', text: res.message },
        ]);
      }
      const detail = await api.plan(slug, tid);
      setPlanDetail(detail);
      refreshTree();
    } catch (err) {
      await alert(err instanceof Error ? err.message : String(err));
    }
  };

  const handleTaskStop = async (taskId: string) => {
    const planThread =
      selected?.kind === 'plan'
        ? selected.thread_id
        : selected?.kind === 'worker'
          ? selected.thread_id.split(':worker:')[0]
          : '';
    if (!slug || !planThread) {
      return;
    }
    try {
      const res = await api.planTaskCancel(slug, planThread, taskId);
      if (!res.ok) {
        await alert(res.message);
      } else {
        setMessages((prev) => [
          ...prev,
          { id: `task-stop-${Date.now()}`, role: 'system', text: res.message },
        ]);
      }
      const detail = await api.plan(slug, planThread);
      setPlanDetail(detail);
      refreshTree();
    } catch (err) {
      await alert(err instanceof Error ? err.message : String(err));
    }
  };

  const handleTaskRun = async (taskId: string) => {
    const planThread =
      selected?.kind === 'plan'
        ? selected.thread_id
        : selected?.kind === 'worker'
          ? selected.thread_id.split(':worker:')[0]
          : '';
    if (!slug || !planThread) {
      return;
    }
    setBusy(true);
    const ac = beginStream(planThread);
    try {
      const check = await api.planTaskRunnable(slug, planThread, taskId);
      if (!check.ok) {
        releasePlanStream(planThread);
        await alert(check.message || '当前 Work 不可执行');
        return;
      }
      await api.planRunTask(
        slug,
        planThread,
        taskId,
        allowWrite,
        bindSSE(planThread),
        ac.signal,
      );
      const detail = await api.plan(slug, planThread);
      setPlanDetail(detail);
      refreshTree();
    } catch (err) {
      releasePlanStream(planThread);
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        await alert(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (!runningThreadsRef.current.has(planThread)) {
        setBusy(false);
      }
    }
  };
  const handlePlanConfirm = async (action: string, allowW: boolean, reviseNote: string) => {
    if (!slug || !selected || selected.kind !== 'plan') {
      return;
    }
    const threadId = selected.thread_id;
    if (allowW) {
      setAllowWrite(true);
    }
    if (action === 'revise' && reviseNote.trim()) {
      setMessages((prev) => [
        ...prev,
        { id: `rev-${Date.now()}`, role: 'user', text: `【修订】${reviseNote.trim()}` },
      ]);
    }
    setBusy(true);
    const ac = beginStream(threadId);
    try {
      await api.planConfirm(
        slug,
        threadId,
        { action, allow_worker_write: allowW, revise_note: reviseNote },
        bindSSE(threadId),
        ac.signal,
      );
      clearConfirmQueue(slug, threadId, 'plan_confirm');
      setPlanConfirm(null);
      const detail = await api.plan(slug, threadId);
      setPlanDetail(detail);
      if (planExecutionAllowWrite(detail)) {
        setAllowWrite(true);
      }
      applyPendingConfirmHead(slug, threadId, {
        setSurvey: () => undefined,
        setPlanConfirm,
        setTaskStepConfirm,
      });
      refreshTree();
    } catch (err) {
      releasePlanStream(threadId);
      try {
        const detail = await api.plan(slug, threadId);
        setPlanDetail(detail);
        if (planNeedsConfirm(detail)) {
          ingestPlanConfirmFromDetail(slug, threadId, detail);
          applyPendingConfirmHead(slug, threadId, {
            setSurvey: () => undefined,
            setPlanConfirm,
            setTaskStepConfirm,
          });
        }
      } catch {
        /* ignore refresh failure */
      }
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        setMessages((prev) => [
          ...prev,
          {
            id: `confirm-err-${Date.now()}`,
            role: 'system',
            text: `**确认失败:** ${err instanceof Error ? err.message : String(err)}`,
          },
        ]);
      }
    } finally {
      if (!runningThreadsRef.current.has(threadId)) {
        setBusy(false);
      }
    }
  };

  const handlePlanContinue = async () => {
    if (!slug || !selected || selected.kind !== 'plan') {
      return;
    }
    const threadId = selected.thread_id;
    const head = peekConfirmHead(slug, threadId);
    const previousTaskStep =
      head?.kind === 'task_step_confirm'
        ? String((head.payload as { task_id?: string })?.task_id || '')
        : '';
    taskStepDismissedRef.current = null;
    setBusy(true);
    const ac = beginStream(threadId);
    try {
      await api.planContinue(slug, threadId, allowWrite, bindSSE(threadId), ac.signal);
      clearConfirmQueue(slug, threadId, 'task_step_confirm');
      setTaskStepConfirm(null);
    } catch (err) {
      releasePlanStream(threadId);
      try {
        const detail = await api.plan(slug, threadId);
        setPlanDetail(detail);
        const pending = detail.plan_state?.pending_interrupt;
        if (pending && typeof pending === 'object' && pending.type === 'task_step_confirm') {
          const taskId = String(pending.task_id || previousTaskStep || '');
          if (taskId) {
            ingestTaskStepConfirm(slug, threadId, taskId);
            applyPendingConfirmHead(slug, threadId, {
              setSurvey: () => undefined,
              setPlanConfirm,
              setTaskStepConfirm,
            });
          }
        } else if (previousTaskStep) {
          ingestTaskStepConfirm(slug, threadId, previousTaskStep);
          setTaskStepConfirm(previousTaskStep);
        }
      } catch {
        if (previousTaskStep) {
          ingestTaskStepConfirm(slug, threadId, previousTaskStep);
          setTaskStepConfirm(previousTaskStep);
        }
      }
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        setMessages((prev) => [
          ...prev,
          {
            id: `continue-err-${Date.now()}`,
            role: 'system',
            text: `**继续执行失败:** ${err instanceof Error ? err.message : String(err)}`,
          },
        ]);
      }
    } finally {
      if (!runningThreadsRef.current.has(threadId)) {
        setBusy(false);
      }
    }
  };

  return {
    maybePromptPlanConfirm,
    handleWorkerSelect,
    handleWorkClick,
    handleBackToPlan,
    handlePlanStop,
    handlePlanAbort,
    handleTaskStop,
    handleTaskRun,
    handlePlanConfirm,
    handlePlanContinue,
  };
}
