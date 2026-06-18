export interface TaskMeta {
  id: string;
  title: string;
  status: string;
  depends_on?: string[];
}

const TERMINAL_STATUSES = new Set(['done', 'skipped']);

export function isDepSatisfied(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export function taskLayers(tasks: TaskMeta[]): TaskMeta[][] {
  if (tasks.length === 0) {
    return [];
  }
  const byId = new Map(tasks.map((t) => [t.id, t]));
  const depthCache = new Map<string, number>();

  const depthOf = (id: string, stack = new Set<string>()): number => {
    if (depthCache.has(id)) {
      return depthCache.get(id)!;
    }
    if (stack.has(id)) {
      return 0;
    }
    stack.add(id);
    const task = byId.get(id);
    const deps = task?.depends_on?.filter((d) => byId.has(d)) || [];
    const d = deps.length === 0 ? 0 : 1 + Math.max(...deps.map((dep) => depthOf(dep, stack)));
    depthCache.set(id, d);
    return d;
  };

  tasks.forEach((t) => depthOf(t.id));
  const maxDepth = Math.max(0, ...tasks.map((t) => depthCache.get(t.id) || 0));
  const layers: TaskMeta[][] = Array.from({ length: maxDepth + 1 }, () => []);
  tasks.forEach((t) => {
    layers[depthCache.get(t.id) || 0].push(t);
  });
  return layers;
}

export function buildWorkEdges(tasks: TaskMeta[]): Array<{ from: string; to: string }> {
  const ids = new Set(tasks.map((t) => t.id));
  const edges: Array<{ from: string; to: string }> = [];
  for (const task of tasks) {
    for (const dep of task.depends_on || []) {
      if (ids.has(dep)) {
        edges.push({ from: dep, to: task.id });
      }
    }
  }
  return edges;
}

export function terminalTaskIds(tasks: TaskMeta[]): string[] {
  const dependedOn = new Set<string>();
  for (const task of tasks) {
    for (const dep of task.depends_on || []) {
      dependedOn.add(dep);
    }
  }
  return tasks.filter((t) => !dependedOn.has(t.id)).map((t) => t.id);
}

export function depsAllSatisfied(task: TaskMeta, byId: Map<string, TaskMeta>): boolean {
  const deps = (task.depends_on || []).filter((d) => byId.has(d));
  if (deps.length === 0) {
    return true;
  }
  return deps.every((dep) => isDepSatisfied(byId.get(dep)?.status || 'pending'));
}

export function isTaskRunnable(task: TaskMeta, byId: Map<string, TaskMeta>): boolean {
  if (task.status === 'running') {
    return false;
  }
  if (isDepSatisfied(task.status)) {
    return false;
  }
  return depsAllSatisfied(task, byId);
}

export function layerHeaderText(layerIndex: number, layerCount: number): { title: string; hint: string } {
  if (layerIndex === 0) {
    return {
      title: `第 1 层 · 可并行`,
      hint: layerCount > 1 ? '无上游依赖，确认后即可调度' : '全部 Work 完成后进入汇总',
    };
  }
  return {
    title: `第 ${layerIndex + 1} 层`,
    hint: '上一层全部完成后，本层 Work 可执行',
  };
}

export function bezierPath(
  from: DOMRect,
  to: DOMRect,
  container: DOMRect,
): string {
  const x1 = from.left + from.width / 2 - container.left;
  const y1 = from.bottom - container.top;
  const x2 = to.left + to.width / 2 - container.left;
  const y2 = to.top - container.top;
  const gap = Math.max(24, (y2 - y1) * 0.45);
  const c1y = y1 + gap;
  const c2y = y2 - gap;
  return `M ${x1} ${y1} C ${x1} ${c1y}, ${x2} ${c2y}, ${x2} ${y2}`;
}
