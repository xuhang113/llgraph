export interface TraceTurn {
  id: string;
  turn_index: number;
  label: string;
  steps: TraceStep[];
  live?: boolean;
}

export function parseTraceTurnsFromRemote(
  turnsRaw: unknown,
  fallbackSteps: TraceStep[] = [],
): TraceTurn[] {
  if (Array.isArray(turnsRaw) && turnsRaw.length > 0) {
    return turnsRaw
      .filter((item): item is Record<string, unknown> => item != null && typeof item === 'object')
      .map((row, index) => {
        const stepsRaw = row.steps;
        const steps: TraceStep[] = Array.isArray(stepsRaw)
          ? stepsRaw
              .filter((item): item is Record<string, unknown> => item != null && typeof item === 'object')
              .map((step, stepIndex) => ({
                step_id: Number(step.step_id ?? stepIndex + 1),
                kind: String(step.kind ?? ''),
                title: String(step.title ?? ''),
                elapsed: Number(step.elapsed ?? 0),
                summary: String(step.summary ?? ''),
                body_lines: Array.isArray(step.body_lines) ? step.body_lines.map(String) : [],
                usage: (step.usage as StepUsage | null | undefined) ?? null,
              }))
          : [];
        const turnIndex = Number(row.turn_index ?? index + 1);
        return {
          id: `turn-${turnIndex}`,
          turn_index: turnIndex,
          label: String(row.label ?? `第 ${turnIndex} 轮`),
          steps,
          live: Boolean(row.live),
        };
      });
  }
  if (fallbackSteps.length > 0) {
    return [{ id: 'turn-1', turn_index: 1, label: '第 1 轮', steps: fallbackSteps }];
  }
  return [];
}

/** 已完成轮次 + 当前轮（live）合并为展示列表。 */
export function buildDisplayTraceTurns(
  completedTurns: TraceTurn[],
  currentSteps: TraceStep[],
  opts: { busy: boolean; currentLabel?: string },
): TraceTurn[] {
  const completed = completedTurns.filter((turn) => !turn.live);
  if (currentSteps.length === 0) {
    return completed;
  }
  const turnIndex = completed.length + 1;
  const label =
    opts.currentLabel ||
    (opts.busy ? `第 ${turnIndex} 轮 · 进行中` : `第 ${turnIndex} 轮`);
  return [
    ...completed,
    {
      id: 'turn-live',
      turn_index: turnIndex,
      label,
      steps: currentSteps,
      live: opts.busy,
    },
  ];
}

export interface StepUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_reported?: boolean;
}

export interface TraceStep {
  step_id: number;
  kind: string;
  title: string;
  elapsed: number;
  summary: string;
  body_lines?: string[];
  usage?: StepUsage | null;
}

export function formatTraceDuration(seconds: number): string {
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)}ms`;
  }
  return `${seconds.toFixed(2)}s`;
}

export function formatTokenAmount(tokens: number): string {
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(2)}M`;
  }
  if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(1)}K`;
  }
  return String(tokens);
}

export function formatStepUsage(usage?: StepUsage | null): string {
  if (!usage) {
    return '';
  }
  const parts: string[] = [];
  const inT = usage.input_tokens ?? 0;
  const outT = usage.output_tokens ?? 0;
  if (inT || outT) {
    parts.push(`token in ${formatTokenAmount(inT)} out ${formatTokenAmount(outT)}`);
  }
  const cacheRead = usage.cache_read_input_tokens ?? 0;
  const cacheCreate = usage.cache_creation_input_tokens ?? 0;
  if (cacheRead || cacheCreate) {
    const bits: string[] = [];
    if (cacheRead) {
      bits.push(`读 ${formatTokenAmount(cacheRead)}`);
    }
    if (cacheCreate) {
      bits.push(`写 ${formatTokenAmount(cacheCreate)}`);
    }
    parts.push(`cache ${bits.join(' ')}`);
  }
  return parts.join(' · ');
}

export function stepMarker(step: TraceStep): string {
  if (step.kind === 'thinking') {
    return '◎';
  }
  if (step.kind === 'reply') {
    return '💬';
  }
  if (step.kind === 'plan' || step.title.includes('模型决策')) {
    return '▶';
  }
  if (step.kind === 'preprocess' || step.kind === 'search_params') {
    return '◇';
  }
  if (step.kind === 'tool' || step.title.startsWith('执行')) {
    return '▷';
  }
  return '▶';
}

const STEP_HEADER_RE = /^\[\d{2}:\d{2}:\d{2}\]\s*(?:▶|▷|◇)\s*#\d+/;
const STEP_DETAIL_RE = /^\s*│/;

/** 从 trace_line 文本中筛出非步骤摘要行（用户消息、思考中、里程碑等）。 */
export interface TraceLineItem {
  id: string;
  text: string;
}

/** 按 step_id 合并步骤，避免 turn_done / live 重复追加。 */
export function mergeTraceStepsUnique(
  panelSteps: TraceStep[],
  incomingSteps: TraceStep[],
): TraceStep[] {
  if (incomingSteps.length === 0) {
    return panelSteps;
  }
  const byId = new Map<number, TraceStep>();
  for (const step of panelSteps) {
    byId.set(step.step_id, step);
  }
  for (const step of incomingSteps) {
    if (!byId.has(step.step_id)) {
      byId.set(step.step_id, step);
    }
  }
  return [...byId.values()].sort((a, b) => a.step_id - b.step_id);
}

/** 无逐行 SSE 时，用步骤摘要合成面板日志行。 */
export function stepsToPanelLogLines(steps: TraceStep[]): TraceLineItem[] {
  return steps.map((step, index) => ({
    id: `syn-${step.step_id}-${index}`,
    text: `${stepMarker(step)} #${index + 1} ${step.title}  ${step.summary}`.trim(),
  }));
}

export function partitionTraceMiscLines(lines: string[]): string[] {
  const misc: string[] = [];
  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }
    if (STEP_HEADER_RE.test(line) || STEP_DETAIL_RE.test(line.trimStart())) {
      continue;
    }
    misc.push(line);
  }
  return misc;
}

/** 有结构化步骤时仅保留用户消息 / 预处理类 misc（隐藏工具里程碑重复日志）。 */
export function filterTraceMiscWhenSteps(miscLines: string[], stepCount: number): string[] {
  if (stepCount <= 0) {
    return miscLines;
  }
  return miscLines.filter((line) => {
    if (line.includes('用户消息')) {
      return true;
    }
    if (/准备中|加载历史|压缩上下文|本轮暂无步骤/.test(line)) {
      return true;
    }
    return false;
  });
}

/** 供贴底滚动依赖：步数 + 最后一步内容变化时更新。 */
export function traceStepsFingerprint(steps: TraceStep[]): string {
  if (steps.length === 0) {
    return '0';
  }
  const last = steps[steps.length - 1]!;
  const bodyLines = steps.reduce((sum, s) => sum + (s.body_lines?.length ?? 0), 0);
  return `${steps.length}:${last.step_id}:${bodyLines}:${last.summary?.length ?? 0}:${last.elapsed}`;
}
