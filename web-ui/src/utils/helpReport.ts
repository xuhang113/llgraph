export interface HelpCommandLine {
  command: string;
  description: string;
}

export interface HelpSection {
  title: string;
  commands: HelpCommandLine[];
  lines: string[];
}

export interface HelpReport {
  title: string;
  sections: HelpSection[];
  footerLines: string[];
}

const SECTION_RE = /^【(.+)】$/;
const COMMAND_RE = /^ {2}(\S+)\s{2,}(.+)$/;
const RULE_RE = /^={3,}$/;

/** 是否为终端 /help 类报告文本。 */
export function isHelpReport(text: string): boolean {
  const t = text.trim();
  return (
    /^llgraph (交互帮助|完整帮助)/m.test(t) ||
    (t.includes('【常用】') && t.includes('【当前会话】'))
  );
}

/** 将 /help 纯文本解析为结构化块（供 Web 卡片展示）。 */
export function parseHelpReport(text: string): HelpReport | null {
  if (!isHelpReport(text)) {
    return null;
  }
  const lines = text.trim().split('\n');
  if (lines.length < 2) {
    return null;
  }
  const title = lines[0].trim();
  const sections: HelpSection[] = [];
  const footerLines: string[] = [];
  let current: HelpSection | null = null;
  let afterSession = false;

  for (let i = 1; i < lines.length; i += 1) {
    const raw = lines[i];
    const line = raw.trimEnd();
    if (!line.trim()) {
      continue;
    }
    if (RULE_RE.test(line.trim())) {
      continue;
    }
    const sectionMatch = line.trim().match(SECTION_RE);
    if (sectionMatch) {
      const secTitle = sectionMatch[1];
      if (secTitle === '当前会话') {
        afterSession = false;
      }
      current = { title: secTitle, commands: [], lines: [] };
      sections.push(current);
      if (secTitle === '当前会话') {
        afterSession = true;
      }
      continue;
    }
    if (line.startsWith('详情:') || (afterSession && !current)) {
      footerLines.push(line.trim());
      continue;
    }
    if (!current) {
      footerLines.push(line.trim());
      continue;
    }
    const cmdMatch = line.match(COMMAND_RE);
    if (cmdMatch) {
      current.commands.push({
        command: cmdMatch[1],
        description: cmdMatch[2].trim(),
      });
      continue;
    }
    if (line.trim().startsWith('  ')) {
      current.lines.push(line.trim());
    } else {
      current.lines.push(line.trim());
    }
  }

  return { title, sections, footerLines };
}
