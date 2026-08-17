/** GFM 表格后若无空行，后续段落会被 remark-gfm 吃进 table。 */
export function ensureBlankLineAfterMarkdownTables(text: string): string {
  const lines = text.split('\n');
  const isTableLine = (line: string) => {
    const t = line.trim();
    return t.startsWith('|') && t.endsWith('|');
  };
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const prev = i > 0 ? lines[i - 1] : '';
    if (line.trim() && !isTableLine(line) && isTableLine(prev)) {
      out.push('');
    }
    out.push(line);
  }
  return out.join('\n');
}

/** 引用块前补空行，避免紧接段落时被当作普通文本。 */
export function ensureBlankLineBeforeBlockquotes(text: string): string {
  const lines = text.split('\n');
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trimStart();
    if (trimmed.startsWith('>') && out.length > 0 && out[out.length - 1]?.trim()) {
      out.push('');
    }
    out.push(line);
  }
  return out.join('\n');
}

/** 跳过 fenced code，只对普通文本做变换。 */
function mapOutsideCodeFences(text: string, transform: (chunk: string) => string): string {
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts
    .map((part) => (part.startsWith('```') ? part : transform(part)))
    .join('');
}

/**
 * 行内 ATX 标题拆行：详解###1. Foo -> 详解\n\n### 1. Foo
 * 模型常把 ### 粘在上一句末尾，标准 MD 无法识别为标题。
 */
export function splitInlineMarkdownHeadings(text: string): string {
  return text
    .split('\n')
    .map((line) => {
      const trimmed = line.trimStart();
      if (!trimmed || trimmed.startsWith('|') || /^-{3,}$/.test(trimmed)) {
        return line;
      }
      // 仅拆 2~6 个 #，避免误伤 #fff、C# 等单 # 场景
      return line.replace(/(?<=[^\n#`])(#{2,6})(?=\S)/g, '\n\n$1');
    })
    .join('\n');
}

/** 行内引用拆行：foo > bar -> foo\n> bar */
export function splitInlineBlockquotes(text: string): string {
  return text
    .split('\n')
    .map((line) => line.replace(/([^\n>])\s+(> .+)$/, '$1\n$2'))
    .join('\n');
}

/** ATX 标题补空格：##标题 / ###1. -> ## 标题 / ### 1. */
export function fixMarkdownHeadings(text: string): string {
  return text.replace(/^(#{1,6})([^\s#\n].*)$/gm, '$1 $2');
}

/** 流式输出时表格行常被 || 粘连，按行拆分。 */
export function fixConcatenatedTableRows(text: string): string {
  return text
    .split('\n')
    .map((line) => {
      if (!line.includes('||') || !line.trimStart().startsWith('|')) {
        return line;
      }
      let row = line;
      let prev = '';
      while (row !== prev) {
        prev = row;
        row = row.replace(/\|\|(?=\s*[-:])/g, '|\n|');
        row = row.replace(/\|\|(?=\s*[^\s|])/g, '|\n|');
      }
      return row;
    })
    .join('\n');
}

function normalizeMarkdownBody(text: string): string {
  return mapOutsideCodeFences(text, (chunk) => {
    let out = splitInlineMarkdownHeadings(chunk);
    out = splitInlineBlockquotes(out);
    out = ensureBlankLineBeforeBlockquotes(out);
    out = fixMarkdownHeadings(out);
    return out;
  });
}

/** 流式尾部可能截断在标题/表格/代码块中间，去掉不完整片段避免破坏渲染。 */
export function trimIncompleteStreamingMarkdown(text: string): string {
  let out = text;
  const fenceMatches = [...out.matchAll(/^```[^\n]*$/gm)];
  if (fenceMatches.length % 2 === 1) {
    const last = fenceMatches[fenceMatches.length - 1];
    if (last.index !== undefined) {
      out = out.slice(0, last.index).trimEnd();
    }
  }

  const lines = out.split('\n');
  while (lines.length > 0) {
    const last = lines[lines.length - 1]?.trim() ?? '';
    if (!last) {
      lines.pop();
      continue;
    }
    if (/^#{1,6}$/.test(last)) {
      lines.pop();
      continue;
    }
    if (/^#{2,6}\s*\S*$/.test(last) && last.length < 24) {
      lines.pop();
      continue;
    }
    if (/^\|/.test(last) && !/\|.*\|/.test(last.slice(1))) {
      lines.pop();
      continue;
    }
    if (/^\|/.test(last) && (last.match(/\|/g)?.length ?? 0) < 2) {
      lines.pop();
      continue;
    }
    break;
  }

  out = lines.join('\n');
  if (!out) {
    return out;
  }

  // 尾部正在输入的行内标题 token：详解### -> 详解
  const tailLines = out.split('\n');
  const lastLine = tailLines[tailLines.length - 1] ?? '';
  if (/(?<=[^\n#`])#{1,6}$/.test(lastLine)) {
    tailLines[tailLines.length - 1] = lastLine.replace(/#{1,6}$/, '').trimEnd();
    out = tailLines.join('\n');
  }

  return out;
}

export function normalizeMarkdownForRender(
  text: string,
  options?: { streaming?: boolean },
): string {
  if (!text.trim()) {
    return text;
  }
  let out = normalizeMarkdownBody(text);
  out = fixConcatenatedTableRows(out);
  out = ensureBlankLineAfterMarkdownTables(out);
  if (options?.streaming) {
    out = trimIncompleteStreamingMarkdown(out);
  }
  return out;
}
