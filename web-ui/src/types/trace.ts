export interface TraceTurn {
  id: string;
  turn_index: number;
  label: string;
  steps: TraceStep[];
  /** 该轮 trace 日志行（含用户消息预览） */
  log_lines?: string[];
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
          ? dedupeTraceStepsById(
              stepsRaw
                .filter((item): item is Record<string, unknown> => item != null && typeof item === 'object')
                .map((step, stepIndex) => ({
                  step_id: Number(step.step_id ?? stepIndex + 1),
                  kind: String(step.kind ?? ''),
                  title: String(step.title ?? ''),
                  elapsed: Number(step.elapsed ?? 0),
                  elapsed_kind: String(step.elapsed_kind ?? resolveElapsedKind(String(step.kind ?? ''))),
                  summary: String(step.summary ?? ''),
                  body_lines: Array.isArray(step.body_lines) ? step.body_lines.map(String) : [],
                  usage: (step.usage as StepUsage | null | undefined) ?? null,
                  invoke_timing:
                    (step.invoke_timing as InvokeTiming | null | undefined) ?? null,
                  sub_thread: step.sub_thread != null ? String(step.sub_thread) : undefined,
                })),
            )
          : [];
        const turnIndex = Number(row.turn_index ?? index + 1);
        const logLinesRaw = row.log_lines;
        const log_lines = Array.isArray(logLinesRaw)
          ? logLinesRaw.map(String).filter((line) => line.trim())
          : undefined;
        return {
          id: `turn-${turnIndex}`,
          turn_index: turnIndex,
          label: String(row.label ?? `第 ${turnIndex} 轮`),
          steps,
          log_lines,
          live: Boolean(row.live),
        };
      });
  }
  if (fallbackSteps.length > 0) {
    return [
      {
        id: 'turn-1',
        turn_index: 1,
        label: '第 1 轮',
        steps: dedupeTraceStepsById(fallbackSteps),
      },
    ];
  }
  return [];
}

/** 已完成轮次 + 当前轮（live）合并为展示列表。 */
export function buildDisplayTraceTurns(
  completedTurns: TraceTurn[],
  currentSteps: TraceStep[],
  opts: { busy: boolean; currentLabel?: string; currentDurationSec?: number },
): TraceTurn[] {
  const completed = completedTurns.filter((turn) => !turn.live);
  if (currentSteps.length === 0) {
    return completed;
  }
  const turnIndex = completed.length + 1;
  let label =
    opts.currentLabel ||
    (opts.busy ? `第 ${turnIndex} 轮 · 进行中` : `第 ${turnIndex} 轮`);
  if (
    !opts.busy &&
    opts.currentDurationSec != null &&
    Number.isFinite(opts.currentDurationSec) &&
    opts.currentDurationSec > 0
  ) {
    label = `第 ${turnIndex} 轮 · ${formatTraceDuration(opts.currentDurationSec)}`;
  }
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

export interface InvokeTiming {
  prepare_sec?: number;
  http_sec?: number;
  request_id?: string;
}

export interface TraceStep {
  step_id: number;
  kind: string;
  title: string;
  elapsed: number;
  summary: string;
  body_lines?: string[];
  usage?: StepUsage | null;
  invoke_timing?: InvokeTiming | null;
  /** model=仅模型 / tool=工具等待 / preprocess=预处理 / wall=墙钟 */
  elapsed_kind?: string;
  /** explore 子会话 thread，展开可拉子 Trace */
  sub_thread?: string | null;
}

export function formatTraceDuration(seconds: number): string {
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)}ms`;
  }
  return `${seconds.toFixed(2)}s`;
}

export function formatStepElapsedLabel(step: TraceStep): string {
  const kind = step.elapsed_kind ?? resolveElapsedKind(step.kind);
  const prefix =
    kind === 'model'
      ? '仅模型'
      : kind === 'tool'
        ? '工具等待'
        : kind === 'preprocess'
          ? '预处理'
          : '耗时';
  return `${prefix} ${formatTraceDuration(step.elapsed)}`;
}

export function resolveElapsedKind(kind: string): string {
  if (kind === 'plan' || kind === 'thinking' || kind === 'reply') {
    return 'model';
  }
  if (kind === 'tool' || kind === 'explore') {
    return 'tool';
  }
  if (
    kind === 'compress' ||
    kind === 'tool_prune' ||
    kind === 'preprocess' ||
    kind === 'search_params' ||
    kind === 'tools' ||
    kind === 'memory_recall' ||
    kind === 'memory_write' ||
    kind === 'memory_consolidate'
  ) {
    return 'preprocess';
  }
  return 'wall';
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

export function formatInvokeTiming(timing?: InvokeTiming | null): string {
  if (!timing) {
    return '';
  }
  const parts: string[] = [];
  const prepare = timing.prepare_sec ?? 0;
  const http = timing.http_sec ?? 0;
  if (prepare > 0.001) {
    parts.push(`prepare ${formatTraceDuration(prepare)}`);
  }
  if (http > 0.001) {
    parts.push(`http ${formatTraceDuration(http)}`);
  }
  const req = (timing.request_id ?? '').trim();
  if (req) {
    parts.push(`req ${req}`);
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
  if (step.kind === 'explore') {
    return '◈';
  }
  if (step.kind === 'plan' || step.title.includes('模型决策')) {
    return '▶';
  }
  if (
    step.kind === 'preprocess' ||
    step.kind === 'search_params' ||
    step.kind === 'tool_prune' ||
    step.kind === 'compress' ||
    step.kind === 'memory_recall' ||
    step.kind === 'memory_write' ||
    step.kind === 'memory_consolidate'
  ) {
    return '◇';
  }
  if (step.kind === 'tool' || step.title.startsWith('执行')) {
    return '▷';
  }
  return '▶';
}

/** invoke 前例行预处理（历史 Trace 中仍可能存在，展示时过滤）。 */
const INVOKE_PRELUDE_TITLES = new Set([
  '工具结果裁剪',
  '上下文检查',
  'Manifest 同步',
  '历史 sanitize',
  '出站上下文',
  'invoke 准备合计',
]);

export function isInvokePreludeStep(step: TraceStep): boolean {
  if (step.kind === 'explore') {
    return false;
  }
  return INVOKE_PRELUDE_TITLES.has(step.title);
}

/** Trace 面板可见步骤：去掉 ReAct 前置例行预处理。 */
export function filterTraceStepsForDisplay(steps: TraceStep[]): TraceStep[] {
  return steps.filter((step) => !isInvokePreludeStep(step));
}

const STEP_HEADER_RE = /^\[\d{2}:\d{2}:\d{2}\]\s*(?:▶|▷|◇)\s*#\d+/;
const STEP_DETAIL_RE = /^\s*│/;

/** 从 trace_line 文本中筛出非步骤摘要行（用户消息、思考中、里程碑等）。 */
export interface TraceLineItem {
  id: string;
  text: string;
}

/** 同 step_id 保留正文更长的版本（Think 回填）。 */
function preferRicherStep(a: TraceStep, b: TraceStep): TraceStep {
  const aBody = (a.body_lines ?? []).join('\n').length;
  const bBody = (b.body_lines ?? []).join('\n').length;
  if (bBody !== aBody) {
    return bBody > aBody ? b : a;
  }
  const aSum = (a.summary ?? '').length;
  const bSum = (b.summary ?? '').length;
  return bSum >= aSum ? b : a;
}

/** 按 step_id 合并步骤；同 id 取更完整正文，避免 Think 回填重复行。 */
export function mergeTraceStepsUnique(
  panelSteps: TraceStep[],
  incomingSteps: TraceStep[],
): TraceStep[] {
  if (incomingSteps.length === 0) {
    return dedupeTraceStepsById(panelSteps);
  }
  const byId = new Map<number, TraceStep>();
  for (const step of panelSteps) {
    const prev = byId.get(step.step_id);
    byId.set(step.step_id, prev ? preferRicherStep(prev, step) : step);
  }
  for (const step of incomingSteps) {
    const prev = byId.get(step.step_id);
    byId.set(step.step_id, prev ? preferRicherStep(prev, step) : step);
  }
  return [...byId.values()].sort((a, b) => a.step_id - b.step_id);
}

/** 数组内按 step_id 去重（保留更完整正文）。 */
export function dedupeTraceStepsById(steps: TraceStep[]): TraceStep[] {
  if (steps.length <= 1) {
    return steps;
  }
  const byId = new Map<number, TraceStep>();
  for (const step of steps) {
    const prev = byId.get(step.step_id);
    byId.set(step.step_id, prev ? preferRicherStep(prev, step) : step);
  }
  if (byId.size === steps.length) {
    return steps;
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
