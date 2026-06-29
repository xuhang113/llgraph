/** 编辑用：去掉列表重名后缀 ` · abc12345` */
export function titleForRenameEdit(displayTitle: string, threadId: string): string {
  let base = displayTitle.trim();
  const suffix = threadId.startsWith('cli-') ? threadId.slice(4) : threadId;
  const tag = ` · ${suffix}`;
  if (base.endsWith(tag)) {
    base = base.slice(0, -tag.length).trimEnd();
  }
  return base;
}

export function resolveSessionFullTitle(
  displayTitle: string,
  titleFull: string | undefined,
  threadId: string,
): string {
  const full = titleFull?.trim();
  if (full) {
    return full;
  }
  return titleForRenameEdit(displayTitle, threadId);
}
