import type { TraceStep, TraceTurn } from '../../types/trace';
import { mergeTraceStepsUnique, stepsToPanelLogLines, parseTraceTurnsFromRemote, buildDisplayTraceTurns } from '../../types/trace';
import { loadTracePanelCache } from '../../utils/tracePanelStore';
import type { ChatMessage } from '../../components/console/ChatThread';
import type { TraceLine } from './types';

export function parseTraceStep(raw: Record<string, unknown>): TraceStep {
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

export function parseTraceSteps(raw: unknown): TraceStep[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .filter((item): item is Record<string, unknown> => item != null && typeof item === 'object')
    .map(parseTraceStep);
}

export function restorePanelTraceFromMessages(msgs: ChatMessage[]): {
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

function panelLineId(text: string, index: number): string {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  return `panel-${index}-${hash >>> 0}`;
}

export function panelLinesFromTexts(texts: string[]): TraceLine[] {
  return texts
    .filter((line) => line.trim())
    .map((text, index) => ({ id: panelLineId(text, index), text }));
}

export function maxStepId(steps: TraceStep[]): number {
  return steps.reduce((max, step) => Math.max(max, step.step_id), 0);
}

export function claimTurnStart(threadId: string, turnOpen: Set<string>): boolean {
  if (!threadId || turnOpen.has(threadId)) {
    return false;
  }
  turnOpen.add(threadId);
  return true;
}

export function releaseTurnOpen(threadId: string, turnOpen: Set<string>): void {
  if (threadId) {
    turnOpen.delete(threadId);
  }
}

function traceLineSeenRecently(lines: TraceLine[], text: string, window = 48): boolean {
  const t = text.trim();
  if (!t) {
    return true;
  }
  return lines.slice(-window).some((line) => line.text.trim() === t);
}

/** 仅在本轮（最后一个「本轮」分隔符之后）去重，避免切回会话时误吞新 trace。 */
export function traceLineSeenInCurrentTurn(lines: TraceLine[], text: string): boolean {
  const t = text.trim();
  if (!t) {
    return true;
  }
  let startIdx = 0;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (lines[i]!.text.trim().startsWith('─── 本轮')) {
      startIdx = i + 1;
      break;
    }
  }
  return lines.slice(startIdx).some((line) => line.text.trim() === t);
}

export function appendUniquePanelLine(lines: TraceLine[], text: string): TraceLine[] {
  if (traceLineSeenRecently(lines, text)) {
    return lines;
  }
  return [...lines, { id: `t-${Date.now()}-${Math.random()}`, text }];
}

export function mergeLiveTraceIntoPanel(
  panelLines: TraceLine[],
  panelSteps: TraceStep[],
  liveLines: TraceLine[],
  liveSteps: TraceStep[],
): { lines: string[]; steps: TraceStep[] } {
  const panelTexts = panelLines.map((line) => line.text);
  const liveTexts = liveLines.map((line) => line.text);
  const mergedSteps = mergeTraceStepsUnique(panelSteps, liveSteps);
  if (liveTexts.length > 0 && panelTexts.length >= liveTexts.length) {
    const tail = panelTexts.slice(-liveTexts.length);
    if (tail.join('\n') === liveTexts.join('\n')) {
      return {
        lines: panelTexts,
        steps: mergedSteps,
      };
    }
  }
  const tailSet = new Set(panelTexts.slice(-64));
  const mergedTexts = [...panelTexts];
  for (const text of liveTexts) {
    if (!tailSet.has(text)) {
      mergedTexts.push(text);
      tailSet.add(text);
    }
  }
  return { lines: mergedTexts, steps: mergedSteps };
}

/** 远端 trace 是否严格新于当前 panel（按 max step id + 行数，避免轮询用旧快照覆盖 SSE）。 */
export function isRemoteTraceAhead(
  remoteSteps: TraceStep[],
  remoteLineCount: number,
  currentSteps: TraceStep[],
  currentLineCount: number,
): boolean {
  const remoteMax = maxStepId(remoteSteps);
  const currentMax = maxStepId(currentSteps);
  if (remoteMax > currentMax) {
    return true;
  }
  if (remoteMax === currentMax && remoteLineCount > currentLineCount) {
    return true;
  }
  return false;
}

export function preferRicherTraceCache(
  slug: string,
  threadId: string,
  panelLines: TraceLine[],
  panelSteps: TraceStep[],
): { lines: TraceLine[]; steps: TraceStep[] } {
  const cached = loadTracePanelCache(slug, threadId);
  if (!cached || (cached.log_lines.length === 0 && cached.steps.length === 0)) {
    return { lines: panelLines, steps: panelSteps };
  }
  const cacheRicher =
    cached.steps.length > panelSteps.length ||
    (cached.steps.length === panelSteps.length && cached.log_lines.length > panelLines.length);
  if (!cacheRicher) {
    return { lines: panelLines, steps: panelSteps };
  }
  return {
    lines: panelLinesFromTexts(cached.log_lines),
    steps: cached.steps,
  };
}

export function pushCompletedTraceTurn(
  turnsRef: { current: TraceTurn[] },
  stepsRef: { current: TraceStep[] },
  setTurns: (turns: TraceTurn[]) => void,
  setSteps: (steps: TraceStep[]) => void,
  label: string,
): void {
  const steps = stepsRef.current;
  if (steps.length === 0) {
    return;
  }
  const turn: TraceTurn = {
    id: `turn-${turnsRef.current.length + 1}-${Date.now()}`,
    turn_index: turnsRef.current.length + 1,
    label,
    steps: [...steps],
  };
  const next = [...turnsRef.current, turn];
  turnsRef.current = next;
  setTurns(next);
  stepsRef.current = [];
  setSteps([]);
}

export function loadTraceTurnsFromRemote(
  remote: { turns?: unknown; steps?: Record<string, unknown>[] },
): { completed: TraceTurn[]; currentSteps: TraceStep[]; live: boolean } {
  const fallbackSteps = parseTraceSteps(remote.steps);
  const parsed = parseTraceTurnsFromRemote(remote.turns, fallbackSteps);
  if (parsed.length === 0) {
    return { completed: [], currentSteps: [], live: false };
  }
  const liveTurn = parsed.find((turn) => turn.live);
  if (liveTurn) {
    const completed = parsed.filter((turn) => !turn.live);
    return { completed, currentSteps: liveTurn.steps, live: true };
  }
  return { completed: parsed, currentSteps: [], live: false };
}

export { buildDisplayTraceTurns, parseTraceTurnsFromRemote };

export function appendPanelTraceTurn(
  linesRef: { current: TraceLine[] },
  stepsRef: { current: TraceStep[] },
  turnSteps: TraceStep[],
  setLines: (v: TraceLine[]) => void,
  setSteps: (v: TraceStep[]) => void,
) {
  if (turnSteps.length === 0) {
    return;
  }
  const prevIds = new Set(stepsRef.current.map((s) => s.step_id));
  const merged = mergeTraceStepsUnique(stepsRef.current, turnSteps);
  const added = merged.filter((s) => !prevIds.has(s.step_id));
  if (added.length === 0) {
    return;
  }
  stepsRef.current = merged;
  setSteps(merged);
  const synth = stepsToPanelLogLines(added);
  const nextLines = [...linesRef.current, ...panelLinesFromTexts(synth.map((l) => l.text))];
  linesRef.current = nextLines;
  setLines(nextLines);
}
