import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  type Capabilities,
  type LlmSettings,
  type MessageItem,
  type PlanDetail,
  type SlashCatalogItem,
  type SurveySpec,
  type TreeNode,
  type Workspace,
} from '../api/client';
import AgentToolbar from '../components/console/AgentToolbar';
import ComposerDock from '../components/console/ComposerDock';
import FileChangesPanel from '../components/console/FileChangesPanel';
import CatalogPanel from '../components/console/CatalogPanel';
import PlanMainPanel from '../components/console/PlanMainPanel';
import WorkerMainPanel from '../components/console/WorkerMainPanel';
import ChatComposer from '../components/console/ChatComposer';
import { isHelpReport } from '../utils/helpReport';
import ChatThread, { type ChatMessage } from '../components/console/ChatThread';
import { buildPlanChatMessages } from '../utils/planChat';
import CursorRightPanel from '../components/console/CursorRightPanel';
import CursorSidebar from '../components/console/CursorSidebar';
import CodeSearchPanel from '../components/console/CodeSearchPanel';
import SurveyDialog, {
  PlanConfirmDialog,
  TaskStepConfirmDialog,
} from '../components/console/SurveyDialogs';
import type { TraceStep } from '../types/trace';
import { stepsToPanelLogLines } from '../types/trace';
import {
  extractMessageContent,
  parseApiMessagesToChat,
} from '../utils/messageText';
import { stripSurveyForDisplay } from '../utils/surveyDisplay';
import { loadTracePanelCache, saveTracePanelCache } from '../utils/tracePanelStore';

const LAST_WORKSPACE_SLUG_KEY = 'llgraph.lastWorkspaceSlug';

interface TraceLine {
  id: string;
  text: string;
}

function buildPlanConfirmPayload(detail: PlanDetail): Record<string, unknown> {
  return {
    type: 'plan_confirm',
    title: detail.title,
    task_count: detail.tasks.length,
    tasks: detail.tasks.map((t) => ({
      id: String(t.id || ''),
      title: String(t.title || t.id || ''),
    })),
  };
}

function parseTraceStep(raw: Record<string, unknown>): TraceStep {
  return {
    step_id: Number(raw.step_id ?? 0),
    kind: String(raw.kind ?? ''),
    title: String(raw.title ?? ''),
    elapsed: Number(raw.elapsed ?? 0),
    summary: String(raw.summary ?? ''),
    body_lines: Array.isArray(raw.body_lines) ? raw.body_lines.map(String) : [],
    usage: (raw.usage as TraceStep['usage']) ?? null,
  };
}

function parseTraceSteps(raw: unknown): TraceStep[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .filter((item): item is Record<string, unknown> => item != null && typeof item === 'object')
    .map(parseTraceStep);
}

function restorePanelTraceFromMessages(msgs: ChatMessage[]): {
  lines: TraceLine[];
  steps: TraceStep[];
} {
  const traces = msgs.filter((m) => m.role === 'trace');
  const last = traces[traces.length - 1];
  if (!last) {
    return { lines: [], steps: [] };
  }
  const lines = last.text
    .split('\n')
    .filter((l) => l.trim())
    .map((text, i) => ({ id: `restore-${i}`, text }));
  const steps = last.traceSteps ?? [];
  if (lines.length === 0 && steps.length > 0) {
    return { lines: stepsToPanelLogLines(steps), steps };
  }
  return { lines, steps };
}

function panelLinesFromTexts(texts: string[]): TraceLine[] {
  return texts
    .filter((line) => line.trim())
    .map((text, index) => ({ id: `panel-${index}-${Math.random()}`, text }));
}

function maxStepId(steps: TraceStep[]): number {
  return steps.reduce((max, step) => Math.max(max, step.step_id), 0);
}

function offsetTraceSteps(steps: TraceStep[], offset: number): TraceStep[] {
  return steps.map((step) => ({ ...step, step_id: offset + step.step_id }));
}

function parseWorkerMessages(data: { messages: MessageItem[]; result: Record<string, unknown> | null }): {
  chat: ChatMessage[];
  traces: TraceLine[];
} {
  const { chat, toolTraces } = parseApiMessagesToChat(data.messages || [], {
    toolToTrace: true,
    idPrefix: 'w',
  });
  if (data.result) {
    chat.push({
      id: 'result',
      role: 'assistant',
      text: '```json\n' + JSON.stringify(data.result, null, 2) + '\n```',
    });
  }
  return { chat, traces: toolTraces };
}

function findPlanNode(plans: TreeNode[], threadId: string): TreeNode | null {
  for (const plan of plans) {
    if (plan.thread_id === threadId) {
      return plan;
    }
  }
  return null;
}

function appendPanelTraceTurn(
  linesRef: { current: TraceLine[] },
  stepsRef: { current: TraceStep[] },
  turnSteps: TraceStep[],
  setLines: (v: TraceLine[]) => void,
  setSteps: (v: TraceStep[]) => void,
) {
  if (turnSteps.length === 0) {
    return;
  }
  const offset = maxStepId(stepsRef.current);
  const adjusted = offsetTraceSteps(turnSteps, offset);
  const nextSteps = [...stepsRef.current, ...adjusted];
  stepsRef.current = nextSteps;
  setSteps(nextSteps);
  const synth = stepsToPanelLogLines(adjusted);
  const nextLines = [...linesRef.current, ...panelLinesFromTexts(synth.map((l) => l.text))];
  linesRef.current = nextLines;
  setLines(nextLines);
}

export default function ConsolePage() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [slug, setSlug] = useState<string>('');
  const [agents, setAgents] = useState<TreeNode[]>([]);
  const [plans, setPlans] = useState<TreeNode[]>([]);
  const [selected, setSelected] = useState<TreeNode | null>(null);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [llmSettings, setLlmSettings] = useState<LlmSettings | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [traceLines, setTraceLines] = useState<TraceLine[]>([]);
  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);
  /** 右侧 Trace 面板：最后一轮完整日志（独立于聊天区） */
  const [panelTraceLines, setPanelTraceLines] = useState<TraceLine[]>([]);
  const [panelTraceSteps, setPanelTraceSteps] = useState<TraceStep[]>([]);
  const [slashCatalog, setSlashCatalog] = useState<SlashCatalogItem[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [allowWrite, setAllowWrite] = useState(false);
  const [planDetail, setPlanDetail] = useState<PlanDetail | null>(null);
  const [survey, setSurvey] = useState<SurveySpec | null>(null);
  const [planConfirm, setPlanConfirm] = useState<Record<string, unknown> | null>(null);
  const [taskStepConfirm, setTaskStepConfirm] = useState<string | null>(null);
  const [fileChangesTick, setFileChangesTick] = useState(0);
  const [streamText, setStreamText] = useState('');
  const [thinkingText, setThinkingText] = useState('');
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
  const [codeSearchOpen, setCodeSearchOpen] = useState(false);
  const [multiSelectMode, setMultiSelectMode] = useState(false);
  const [selectedSessionIds, setSelectedSessionIds] = useState<Set<string>>(() => new Set());
  const streamedRef = useRef(false);
  const traceFlushedRef = useRef(false);
  const traceStepsRef = useRef<TraceStep[]>([]);
  const traceLinesRef = useRef<TraceLine[]>([]);
  const panelTraceLinesRef = useRef<TraceLine[]>([]);
  const panelTraceStepsRef = useRef<TraceStep[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<number | null>(null);
  const confirmPromptedRef = useRef<string | null>(null);
  const surveyDismissedRef = useRef<string | null>(null);

  const refreshWorkspaces = useCallback(() => {
    api.workspaces().then((d) => setWorkspaces(d.workspaces));
  }, []);

  const refreshTree = useCallback(() => {
    if (!slug) {
      return;
    }
    api.tree(slug).then((t) => {
      setAgents(t.agents);
      setPlans(t.plans);
    });
    api.capabilities(slug, allowWrite).then(setCaps);
    api.llmSettings(slug).then(setLlmSettings).catch(() => setLlmSettings(null));
    api.slashCatalog(slug).then((r) => setSlashCatalog(r.items)).catch(() => setSlashCatalog([]));
  }, [slug, allowWrite]);

  const refreshCaps = useCallback(() => {
    if (!slug) {
      return;
    }
    api.capabilities(slug, allowWrite).then(setCaps);
  }, [slug, allowWrite]);

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

  useEffect(() => {
    refreshWorkspaces();
  }, [refreshWorkspaces]);

  useEffect(() => {
    if (slug || workspaces.length === 0) {
      return;
    }
    const saved = localStorage.getItem(LAST_WORKSPACE_SLUG_KEY);
    if (saved && workspaces.some((w) => w.slug === saved)) {
      setSlug(saved);
      return;
    }
    setSlug(workspaces[0].slug);
  }, [workspaces, slug]);

  useEffect(() => {
    if (slug) {
      localStorage.setItem(LAST_WORKSPACE_SLUG_KEY, slug);
    }
  }, [slug]);

  useEffect(() => {
    if (!slug) {
      return;
    }
    api.touchWorkspace(slug).then(() => refreshWorkspaces()).catch(() => {});
  }, [slug, refreshWorkspaces]);

  useEffect(() => {
    refreshTree();
  }, [refreshTree]);

  const maybePromptPlanConfirm = useCallback((detail: PlanDetail, threadId: string) => {
    if (detail.phase !== 'awaiting_confirm') {
      return;
    }
    if (confirmPromptedRef.current === threadId) {
      return;
    }
    confirmPromptedRef.current = threadId;
    setPlanConfirm(buildPlanConfirmPayload(detail));
  }, []);

  useEffect(() => {
    if (selected?.kind === 'plan' && planDetail?.phase === 'awaiting_confirm') {
      maybePromptPlanConfirm(planDetail, selected.thread_id);
    }
    if (selected?.kind === 'plan' && planDetail?.phase !== 'awaiting_confirm') {
      confirmPromptedRef.current = null;
    }
  }, [selected, planDetail, maybePromptPlanConfirm]);

  useEffect(() => {
    if (!slug || selected?.kind !== 'plan') {
      return;
    }
    const planThread = selected.thread_id;
    const refreshPlan = () => {
      api.plan(slug, planThread).then(setPlanDetail).catch(() => {});
      refreshTree();
    };
    const unsub = api.subscribePlanEvents(slug, planThread, (ev) => {
      const t = String(ev.type || '');
      if (t === 'plan_state' || t === 'turn_done' || t === 'end' || t === 'subscribed') {
        refreshPlan();
      }
    });
    return unsub;
  }, [slug, selected?.kind, selected?.thread_id, refreshTree]);

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
        window.alert('请先确认计划后再执行 Work');
      } else {
        handleWorkerSelect(taskId);
      }
      return;
    }

    if (status === 'done') {
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
        window.alert(check.message || '当前 Work 不可执行');
        return;
      }
      setBusy(true);
      abortRef.current = new AbortController();
      await api.planRunTask(
        slug,
        selected.thread_id,
        taskId,
        allowWrite,
        onSSE,
        abortRef.current.signal,
      );
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
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

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamText, traceLines, traceSteps, busy]);

  useEffect(() => {
    try {
      localStorage.setItem('llgraph-right-panel', rightPanelOpen ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [rightPanelOpen]);

  const loadHistory = useCallback(
    async (node: TreeNode) => {
      if (!slug) {
        return;
      }
      setMessages([]);
      setTraceLines([]);
      setTraceSteps([]);
      setPanelTraceLines([]);
      setPanelTraceSteps([]);
      setPlanDetail(null);
      setSurvey(null);
      surveyDismissedRef.current = null;
      if (node.kind === 'agent') {
        let data: { messages: MessageItem[] };
        try {
          data = await api.messages(slug, node.thread_id);
        } catch (err) {
          setMessages([
            {
              id: 'load-err',
              role: 'system',
              text: `**加载历史失败:** ${err instanceof Error ? err.message : String(err)}`,
            },
          ]);
          return;
        }
        const parsed = parseApiMessagesToChat(data.messages).chat;
        setMessages(parsed);
        let panelLines: TraceLine[] = [];
        let panelSteps: TraceStep[] = [];
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
          /* 无落盘 trace 时走缓存 */
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
        setPanelTraceLines(panelLines);
        setPanelTraceSteps(panelSteps);
        panelTraceLinesRef.current = panelLines;
        panelTraceStepsRef.current = panelSteps;
      } else if (node.kind === 'plan') {
        const detail = await api.plan(slug, node.thread_id);
        setPlanDetail(detail);
        maybePromptPlanConfirm(detail, node.thread_id);
        let history: MessageItem[] = [];
        try {
          const data = await api.messages(slug, node.thread_id);
          history = data.messages || [];
        } catch {
          /* plan 会话可能尚无 messages.jsonl */
        }
        setMessages(buildPlanChatMessages(detail, history));
        let panelLines: TraceLine[] = [];
        let panelSteps: TraceStep[] = [];
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
        if (panelLines.length === 0 && panelSteps.length === 0) {
          const cached = loadTracePanelCache(slug, node.thread_id);
          if (cached) {
            panelLines = panelLinesFromTexts(cached.log_lines);
            panelSteps = cached.steps;
          }
        }
        setPanelTraceLines(panelLines);
        setPanelTraceSteps(panelSteps);
        panelTraceLinesRef.current = panelLines;
        panelTraceStepsRef.current = panelSteps;
      } else if (node.kind === 'worker') {
        const planThread = node.thread_id.split(':worker:')[0];
        const taskId = node.task_id || '';
        const [data, detail] = await Promise.all([
          api.worker(slug, planThread, taskId),
          api.plan(slug, planThread),
        ]);
        setPlanDetail(detail);
        const { chat, traces } = parseWorkerMessages(data);
        setMessages(chat);
        setTraceLines(traces);
        setPanelTraceLines(traces);
        setPanelTraceSteps([]);
      }
    },
    [slug, maybePromptPlanConfirm],
  );

  const handleSelect = (node: TreeNode) => {
    setCatalogOpen(null);
    setSelected(node);
    loadHistory(node);
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (node.kind === 'plan' && slug) {
      // Plan 状态由 SSE 订阅更新；此处仅保留慢轮询兜底
      pollRef.current = window.setInterval(async () => {
        try {
          const detail = await api.plan(slug, node.thread_id);
          setPlanDetail(detail);
        } catch {
          /* ignore */
        }
      }, 12000);
    } else if (node.kind === 'worker' && slug) {
      const planThread = node.thread_id.split(':worker:')[0];
      const taskId = node.task_id || '';
      pollRef.current = window.setInterval(async () => {
        try {
          const [data, detail] = await Promise.all([
            api.worker(slug, planThread, taskId),
            api.plan(slug, planThread),
          ]);
          setPlanDetail(detail);
          const { chat, traces } = parseWorkerMessages(data);
          setMessages(chat);
          setTraceLines(traces);
          setPanelTraceLines(traces);
          refreshTree();
        } catch {
          /* ignore */
        }
      }, 3000);
    }
  };

  useEffect(
    () => () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
      }
    },
    [],
  );

  const ensureWorkspace = (): string | null => {
    if (slug) {
      return slug;
    }
    if (workspaces.length > 0) {
      const first = workspaces[0].slug;
      setSlug(first);
      return first;
    }
    window.alert('请先选择工作区，或点击「打开新工作空间」');
    return null;
  };

  const handleOpenWorkspace = async () => {
    try {
      const picked = await api.pickWorkspaceDirectory('');
      if (picked.cancelled || !picked.path) {
        return;
      }
      const ws = await api.registerWorkspace(picked.path);
      setSlug(ws.slug);
      refreshWorkspaces();
      refreshTree();
    } catch (err) {
      window.alert(err instanceof Error ? err.message : '打开工作区失败');
    }
  };

  const handleDismissWorkspace = async (dismissSlug: string) => {
    const ws = workspaces.find((w) => w.slug === dismissSlug);
    const label = ws?.path.split('/').filter(Boolean).pop() || dismissSlug;
    if (
      !window.confirm(
        `从最近列表移除「${label}」？\n\n仅隐藏侧栏入口，Agent/Plan 会话数据仍保留；再次打开同一路径可恢复。`,
      )
    ) {
      return;
    }
    try {
      await api.dismissWorkspace(dismissSlug);
      const nextList = workspaces.filter((w) => w.slug !== dismissSlug);
      if (slug === dismissSlug) {
        const nextSlug = nextList[0]?.slug || '';
        setSlug(nextSlug);
        setSelected(null);
        setCatalogOpen(null);
        setMessages([]);
        setPlanDetail(null);
        if (nextSlug) {
          refreshTree();
        }
      }
      refreshWorkspaces();
    } catch (err) {
      window.alert(err instanceof Error ? err.message : '移除失败');
    }
  };

  const handleNewAgent = async () => {
    const activeSlug = ensureWorkspace();
    if (!activeSlug) {
      return;
    }
    const { thread_id } = await api.createSession(activeSlug, 'agent');
    refreshTree();
    handleSelect({ kind: 'agent', thread_id, title: '新会话', children: [] });
  };

  const handleNewPlan = async () => {
    const activeSlug = ensureWorkspace();
    if (!activeSlug) {
      return;
    }
    const goal = window.prompt('Plan 目标（自然语言）') || '';
    const { thread_id } = await api.createSession(activeSlug, 'plan', goal);
    refreshTree();
    const node: TreeNode = {
      kind: 'plan',
      thread_id,
      title: goal || thread_id,
      children: [],
    };
    handleSelect(node);
    if (goal.trim()) {
      await sendPlanMessage(node, goal);
    }
  };

  const handleCatalogOpen = (kind: 'skills' | 'rules' | 'tools') => {
    const activeSlug = ensureWorkspace();
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
    if (!window.confirm(`确定删除 ${label}？此操作不可恢复。`)) {
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
      window.alert(err instanceof Error ? err.message : '删除失败');
    }
  };

  const handleRenameSession = async (node: TreeNode, title: string) => {
    if (!slug) {
      throw new Error('未选择工作区');
    }
    const res = await api.renameSession(slug, node.thread_id, title);
    if (selected?.thread_id === node.thread_id) {
      setSelected({ ...selected, title: res.title });
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
      !window.confirm(
        `确定删除选中的 ${ids.length} 个会话？Plan 将级联删除 Worker。此操作不可恢复。`,
      )
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
        window.alert(`已删除 ${res.success_count} 个，失败 ${res.failure_count} 个\n${failed}`);
      }
    } catch (err) {
      window.alert(err instanceof Error ? err.message : '批量删除失败');
    }
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
  };

  const onSSE = (event: Record<string, unknown>) => {
    const type = String(event.type || '');
    if (type === 'turn_start') {
      setThinkingText('');
      const entry = {
        id: `turn-sep-${Date.now()}`,
        text: `─── 本轮 ${new Date().toLocaleTimeString('zh-CN', { hour12: false })} ───`,
      };
      const nextLines = [...panelTraceLinesRef.current, entry];
      panelTraceLinesRef.current = nextLines;
      setPanelTraceLines(nextLines);
    } else if (type === 'trace_line') {
      const line = String(event.text || '');
      if (line.trim()) {
        const entry = { id: `t-${Date.now()}-${Math.random()}`, text: line };
        setTraceLines((prev) => {
          const next = [...prev, entry];
          traceLinesRef.current = next;
          return next;
        });
        setPanelTraceLines((prev) => {
          const next = [...prev, entry];
          panelTraceLinesRef.current = next;
          return next;
        });
      }
    } else if (type === 'thinking_delta') {
      const text = String(event.text || '');
      if (text.trim()) {
        setThinkingText(text);
      }
    } else if (type === 'stream_delta') {
      streamedRef.current = true;
      setStreamText((s) => s + String(event.text || ''));
    } else if (type === 'stream_end') {
      finalizeLiveTrace(selected?.thread_id);
      setStreamText((s) => {
        const text = stripSurveyForDisplay(extractMessageContent(s));
        if (text.trim()) {
          streamedRef.current = true;
          setMessages((prev) => [
            ...prev,
            { id: `a-${Date.now()}`, role: 'assistant', text },
          ]);
        }
        return '';
      });
    } else if (type === 'trace_step') {
      const raw = event.step as Record<string, unknown> | undefined;
      if (raw) {
        const step = parseTraceStep(raw);
        const next = [...traceStepsRef.current, step];
        traceStepsRef.current = next;
        setTraceSteps(next);
      }
    } else if (type === 'turn_done' || type === 'survey') {
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
      if (slug && selected?.thread_id) {
        saveTracePanelCache(
          slug,
          selected.thread_id,
          panelTraceLinesRef.current.map((line) => line.text),
          panelTraceStepsRef.current,
        );
      }
      finalizeLiveTrace(selected?.thread_id);
      const text = stripSurveyForDisplay(extractMessageContent(event.text));
      if (text.trim() && !streamedRef.current) {
        setMessages((prev) => [...prev, { id: `done-${Date.now()}`, role: 'assistant', text }]);
      }
      streamedRef.current = false;
      if (event.survey) {
        setSurvey(event.survey as SurveySpec);
      }
      if (selected?.kind === 'agent' || selected?.kind === 'plan' || selected?.kind === 'worker') {
        setFileChangesTick((n) => n + 1);
      }
    } else if (type === 'interrupt') {
      const payload = event.payload as Record<string, unknown>;
      if (payload?.type === 'plan_confirm') {
        setPlanConfirm(payload);
      } else if (payload?.type === 'task_step_confirm') {
        setTaskStepConfirm(String(payload.task_id || ''));
      } else if (payload?.type === 'tasks_incomplete') {
        setMessages((prev) => [
          ...prev,
          {
            id: `intr-${Date.now()}`,
            role: 'system',
            text: String(payload.message || '仍有未完成任务，可在 Plan 面板继续执行'),
          },
        ]);
      }
    } else if (type === 'plan_job') {
      const running = Boolean(event.running);
      const jobError = event.error != null ? String(event.error) : null;
      setPlanDetail((prev) =>
        prev ? { ...prev, job: { running, error: jobError } } : prev,
      );
    } else if (type === 'plan_done') {
      setFileChangesTick((n) => n + 1);
      if (slug && selected?.kind === 'plan') {
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
    } else if (type === 'plan_state') {
      if (selected?.kind === 'plan' && slug) {
        api.plan(slug, selected.thread_id).then(setPlanDetail);
      }
    } else if (type === 'error') {
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'system', text: `**错误:** ${event.message}` },
      ]);
      if (selected?.kind === 'plan' && slug) {
        api.plan(slug, selected.thread_id).then(setPlanDetail).catch(() => {});
      }
    } else if (type === 'end') {
      finalizeLiveTrace(selected?.thread_id);
      setBusy(false);
      streamedRef.current = false;
      refreshTree();
    }
  };

  const liveTraceText = traceLines.map((l) => l.text).join('\n');
  const displayPanelSteps =
    busy && traceSteps.length > 0
      ? [...panelTraceSteps, ...offsetTraceSteps(traceSteps, maxStepId(panelTraceSteps))]
      : panelTraceSteps;
  const displayPanelLines =
    busy && traceLines.length > 0
      ? [...panelTraceLines, ...panelLinesFromTexts(traceLines.map((l) => l.text))]
      : panelTraceLines;

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

  const handleWebSearchChange = async (enabled: boolean) => {
    if (!slug) {
      return;
    }
    const threadId = selected?.kind === 'agent' ? selected.thread_id : '';
    const res = await api.setWebSearch(slug, enabled, threadId, allowWrite);
    setCaps((prev) => (prev ? { ...prev, web_search_enabled: res.enabled } : prev));
  };

  const handleSandboxChange = async (enabled: boolean) => {
    if (!slug) {
      return;
    }
    const threadId = selected?.kind === 'agent' ? selected.thread_id : '';
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
    window.alert(res.message);
    refreshTree();
    refreshWorkspaces();
  };

  const handleStop = () => {
    abortRef.current?.abort();
    finalizeLiveTrace(selected?.thread_id);
    setBusy(false);
    setStreamText('');
    setMessages((prev) => [
      ...prev,
      { id: `stop-${Date.now()}`, role: 'system', text: '已停止当前生成。' },
    ]);
  };

  const sendAgentMessage = async (node: TreeNode, text: string) => {
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
    abortRef.current = new AbortController();
    try {
      await api.agentChat(slug, node.thread_id, text, allowWrite, onSSE, abortRef.current.signal);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        setBusy(false);
        return;
      }
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'system', text: String(e) },
      ]);
      setBusy(false);
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
    abortRef.current = new AbortController();
    try {
      if (planDetail?.phase === 'completed') {
        await api.planDiscuss(slug, node.thread_id, text, onSSE, abortRef.current.signal);
      } else {
        await api.planStart(slug, node.thread_id, text, allowWrite, onSSE, abortRef.current.signal);
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        setBusy(false);
        return;
      }
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: 'system', text: String(e) },
      ]);
      setBusy(false);
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || !selected || busy) {
      return;
    }
    setInput('');

    if (text.startsWith('/') && slug) {
      setBusy(true);
      try {
        const res = await api.metaCommand(slug, text, allowWrite, selected.thread_id);
        if (res.registered) {
          setMessages((prev) => [...prev, { id: `u-${Date.now()}`, role: 'user', text }]);
          if (res.output.trim()) {
            setMessages((prev) => [
              ...prev,
              {
                id: `meta-${Date.now()}`,
                role: 'system',
                text: res.output,
                banner: isHelpReport(res.output) ? 'help' : 'default',
              },
            ]);
          }
          refreshTree();
          setBusy(false);
          return;
        }
      } catch (e) {
        setMessages((prev) => [
          ...prev,
          { id: `err-${Date.now()}`, role: 'system', text: String(e) },
        ]);
        setBusy(false);
        return;
      }
      setBusy(false);
    }

    if (selected.kind === 'agent') {
      await sendAgentMessage(selected, text);
    } else if (selected.kind === 'plan') {
      await sendPlanMessage(selected, text);
    }
  };

  const handlePlanConfirm = async (action: string, allowW: boolean, reviseNote: string) => {
    if (!slug || !selected || selected.kind !== 'plan') {
      return;
    }
    setPlanConfirm(null);
    if (action === 'revise' && reviseNote.trim()) {
      setMessages((prev) => [
        ...prev,
        { id: `rev-${Date.now()}`, role: 'user', text: `【修订】${reviseNote.trim()}` },
      ]);
    }
    setBusy(true);
    abortRef.current = new AbortController();
    try {
      await api.planConfirm(
        slug,
        selected.thread_id,
        { action, allow_worker_write: allowW, revise_note: reviseNote },
        onSSE,
        abortRef.current.signal,
      );
    } finally {
      setBusy(false);
    }
  };

  const handlePlanContinue = async () => {
    if (!slug || !selected || selected.kind !== 'plan') {
      return;
    }
    setBusy(true);
    abortRef.current = new AbortController();
    try {
      await api.planContinue(slug, selected.thread_id, allowWrite, onSSE, abortRef.current.signal);
    } finally {
      setBusy(false);
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
    };
  }, [selected, planDetail]);

  const workerTraceText = traceLines.map((l) => l.text).join('\n');

  const canChat = selected && (selected.kind === 'agent' || selected.kind === 'plan');
  const showPlanMain = selected?.kind === 'plan' && planDetail;
  const showWorkerMain = selected?.kind === 'worker' && workerMeta;

  return (
    <div
      className={`cursor-layout${
        selected && !catalogOpen && rightPanelOpen ? '' : ' cursor-layout--right-collapsed'
      }`}
    >
      <CursorSidebar
        slug={slug}
        workspaces={workspaces}
        agents={agents}
        plans={plans}
        selectedId={selected?.thread_id || null}
        catalogOpen={catalogOpen}
        multiSelectMode={multiSelectMode}
        selectedSessionIds={selectedSessionIds}
        busy={busy}
        onSlugChange={(s) => {
          setSlug(s);
          setSelected(null);
          setCatalogOpen(null);
          setMultiSelectMode(false);
          setSelectedSessionIds(new Set());
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
                <h1 className="cursor-session-title">{sessionTitle}</h1>
                {selected.kind === 'agent' && (
                  <AgentToolbar
                    llm={llmSettings}
                    busy={busy}
                    isAgent
                    webSearchEnabled={caps?.web_search_enabled ?? false}
                    onModelChange={handleModelChange}
                    onThinkingChange={handleThinkingChange}
                    onWebSearchChange={handleWebSearchChange}
                  />
                )}
              </div>
              <div className="cursor-main-actions">
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
                {(selected.kind === 'agent' || selected.kind === 'plan') && caps?.sandbox && (
                  <label
                    className="cursor-toggle"
                    title={
                      caps.sandbox.enabled
                        ? `OS 沙箱已启用 · ${caps.sandbox.backend || 'unknown'} · ${caps.sandbox.mode}`
                        : caps.sandbox.active
                          ? '沙箱已请求但未就绪（后端不可用）'
                          : '启用 OS 沙箱（macOS sandbox-exec / Linux bwrap）'
                    }
                  >
                    <input
                      type="checkbox"
                      checked={caps.sandbox.enabled}
                      disabled={busy}
                      onChange={(e) => void handleSandboxChange(e.target.checked)}
                    />
                    沙箱
                  </label>
                )}
                {selected.kind === 'plan' && (
                  <label className="cursor-toggle" title="联网搜索（Tavily web_search）">
                    <input
                      type="checkbox"
                      checked={caps?.web_search_enabled ?? false}
                      disabled={busy}
                      onChange={(e) => void handleWebSearchChange(e.target.checked)}
                    />
                    联网
                  </label>
                )}
                {(selected.kind === 'agent' || selected.kind === 'plan') && (
                  <label className="cursor-toggle">
                    <input
                      type="checkbox"
                      checked={allowWrite}
                      onChange={(e) => setAllowWrite(e.target.checked)}
                    />
                    允许写
                  </label>
                )}
              </div>
            </header>

            <div className="cursor-main-scroll">
              {selected?.kind === 'plan' && !planDetail ? (
                <div className="cursor-catalog-empty">加载 Plan 工作流…</div>
              ) : showPlanMain ? (
                <>
                  <PlanMainPanel
                    slug={slug}
                    planDetail={planDetail}
                    busy={busy}
                    onTaskSelect={handleWorkClick}
                    onPlanConfirm={() =>
                      planDetail && setPlanConfirm(buildPlanConfirmPayload(planDetail))
                    }
                    onPlanContinue={handlePlanContinue}
                  />
                  <section className="cursor-plan-chat">
                    <header className="cursor-plan-chat-header">
                      <h2 className="cursor-plan-chat-title">对话与修订</h2>
                      <p className="cursor-plan-chat-hint">
                        目标与修订说明在此；Planner 调研过程见右侧 Trace 面板
                      </p>
                    </header>
                    <ChatThread
                      messages={messages}
                      liveTraceText={liveTraceText}
                      liveTraceSteps={traceSteps}
                      liveThinkingText={thinkingText}
                      streamText={streamText}
                      busy={busy}
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
                  messages={messages}
                  traceLines={workerTraceText}
                  busy={busy || workerMeta.status === 'running'}
                  onBack={handleBackToPlan}
                />
              ) : (
                <ChatThread
                  messages={messages}
                  liveTraceText={liveTraceText}
                  liveTraceSteps={traceSteps}
                  liveThinkingText={thinkingText}
                  streamText={streamText}
                  busy={busy}
                  traceMode={caps?.trace_mode}
                />
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
              className="cursor-panel-rail"
              onClick={() => setRightPanelOpen(true)}
              title="展开 Trace / Tools 面板"
              aria-label="展开右侧面板"
            >
              ◂
            </button>
          )}
          <div className={`cursor-right-wrap${rightPanelOpen ? '' : ' is-collapsed'}`}>
            <button
              type="button"
              className="cursor-right-collapse"
              onClick={() => setRightPanelOpen(false)}
              title="收起面板"
              aria-label="收起右侧面板"
            >
              ▸
            </button>
            <CursorRightPanel
              caps={caps}
              traceLines={displayPanelLines}
              traceSteps={displayPanelSteps}
              liveThinking={thinkingText}
              planDetail={selected.kind === 'plan' ? planDetail : null}
              slug={slug}
              threadId={
                selected.kind === 'worker'
                  ? selected.thread_id.split(':worker:')[0]
                  : selected.thread_id
              }
              isPlan={selected.kind === 'plan'}
              isAgent={selected.kind === 'agent'}
              allowWrite={allowWrite}
              onMetaOutput={(text) => {
                setMessages((prev) => [
                  ...prev,
                  {
                    id: `meta-${Date.now()}`,
                    role: 'system',
                    text,
                    banner: isHelpReport(text) ? 'help' : 'default',
                  },
                ]);
              }}
              onTraceMode={(m) => {
                if (!slug) {
                  return;
                }
                api.setTraceMode(slug, m).then((r) => {
                  setCaps((prev) => (prev ? { ...prev, trace_mode: r.mode } : prev));
                });
              }}
              onPlanConfirm={() =>
                planDetail && setPlanConfirm(buildPlanConfirmPayload(planDetail))
              }
              onPlanContinue={handlePlanContinue}
              busy={busy}
              onTaskSelect={handleWorkClick}
            />
          </div>
        </>
      )}

      {survey && slug && (
        <SurveyDialog
          survey={survey}
          onCancel={() => {
            const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
            if (lastAssistant) {
              surveyDismissedRef.current = lastAssistant.id;
            }
            setSurvey(null);
          }}
          onSubmit={async (answers) => {
            setSurvey(null);
            try {
              const { message } = await api.formatSurveyAnswers(slug, answers, allowWrite);
              if (selected?.kind === 'agent') {
                sendAgentMessage(selected, message);
              } else if (selected?.kind === 'plan') {
                sendPlanMessage(selected, message);
              }
            } catch (err) {
              window.alert(err instanceof Error ? err.message : String(err));
            }
          }}
        />
      )}

      {taskStepConfirm && (
        <TaskStepConfirmDialog
          taskId={taskStepConfirm}
          onDismiss={() => setTaskStepConfirm(null)}
          onContinue={() => {
            setTaskStepConfirm(null);
            handlePlanContinue();
          }}
        />
      )}

      {planConfirm && (
        <PlanConfirmDialog
          payload={planConfirm}
          onCancel={() => setPlanConfirm(null)}
          onConfirm={handlePlanConfirm}
        />
      )}

      {codeSearchOpen && slug && (
        <CodeSearchPanel slug={slug} onClose={() => setCodeSearchOpen(false)} />
      )}
    </div>
  );
}
