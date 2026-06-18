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

/** 无逐行 SSE 时，用步骤摘要合成面板日志行。 */
export function stepsToPanelLogLines(steps: TraceStep[]): TraceLineItem[] {
  return steps.map((step) => ({
    id: `syn-${step.step_id}`,
    text: `${stepMarker(step)} #${step.step_id} ${step.title}  ${step.summary}`.trim(),
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
