import type { SlashCatalogItem } from '../api/client';

/** 解析可补全的斜杠前缀（光标前无空格）。 */
export function parseSlashPartial(line: string, cursor: number): string | null {
  if (!line.startsWith('/')) {
    return null;
  }
  const before = line.slice(0, Math.max(0, cursor));
  if (!before.startsWith('/')) {
    return null;
  }
  const rest = before.slice(1);
  if (rest.includes(' ')) {
    return null;
  }
  return rest;
}

/** 按首 token 前缀过滤（与终端 filter_slash_catalog 一致）。 */
export function filterSlashCatalog(
  catalog: SlashCatalogItem[],
  partial: string,
  limit = 16,
): SlashCatalogItem[] {
  const key = partial.replace(/^\//, '').toLowerCase();
  const matched = key
    ? catalog.filter((item) => item.name.toLowerCase().startsWith(key))
    : [...catalog];
  const order: Record<string, number> = { Skills: 0, Commands: 1, 内置: 2 };
  matched.sort(
    (a, b) =>
      (order[a.category] ?? 9) - (order[b.category] ?? 9) ||
      a.name.localeCompare(b.name),
  );
  return matched.slice(0, limit);
}
