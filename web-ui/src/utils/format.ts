export function formatTime(iso: string | null | undefined): string {
  if (!iso) {
    return '—';
  }
  try {
    return new Date(iso).toLocaleString('zh-CN');
  } catch {
    return iso;
  }
}

export function phaseBadgeClass(phase: string): string {
  if (phase.includes('completed')) {
    return 'badge badge-success';
  }
  if (phase.includes('executing') || phase.includes('planning')) {
    return 'badge badge-running';
  }
  if (phase.includes('awaiting')) {
    return 'badge badge-waiting';
  }
  if (phase.includes('cancel') || phase.includes('fail')) {
    return 'badge badge-error';
  }
  return 'badge';
}

export function contentToText(content: unknown): string {
  if (content == null) {
    return '';
  }
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === 'string') {
          return block;
        }
        if (block && typeof block === 'object' && 'text' in block) {
          return String((block as { text: string }).text);
        }
        return JSON.stringify(block);
      })
      .join('\n');
  }
  return JSON.stringify(content, null, 2);
}
