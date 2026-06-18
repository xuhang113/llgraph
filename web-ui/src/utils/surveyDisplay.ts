/** 与后端 survey_prompt.py 对齐：对话区隐藏 survey JSON / 确认列表 */

const SURVEY_START = '<<<llgraph-survey>>>';
const SURVEY_END = '<<<end-survey>>>';
const SURVEY_BLOCK_RE = new RegExp(
  `${escapeRegExp(SURVEY_START)}[\\s\\S]*?(?:${escapeRegExp(SURVEY_END)}|$)`,
  'g',
);
const CONFIRMATION_HEADER_RE = /(请确认|确认你的需求|确认以下|请选择以下|请在下方确认)/i;
const NUMBERED_OPTION_RE = /^\s*(\d+)[.)、]\s*(?:\*\*)?(.+?)(?:\*\*)?\s*$/;

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stripConfirmationMarkdown(text: string): string {
  if (!CONFIRMATION_HEADER_RE.test(text)) {
    return text;
  }
  const lines = text.split('\n');
  const out: string[] = [];
  let skipping = false;
  for (const line of lines) {
    if (!skipping && CONFIRMATION_HEADER_RE.test(line)) {
      const match = CONFIRMATION_HEADER_RE.exec(line);
      if (match && match.index > 0) {
        const prefix = line.slice(0, match.index).trim();
        if (prefix) {
          out.push(prefix);
        }
      }
      skipping = true;
      continue;
    }
    if (skipping) {
      if (NUMBERED_OPTION_RE.test(line.trim())) {
        continue;
      }
      if (line.trim() === '') {
        continue;
      }
      if (line.trim().startsWith('---')) {
        skipping = false;
        out.push(line);
        continue;
      }
      skipping = false;
    }
    out.push(line);
  }
  return out.join('\n').trim();
}

/** 从展示用正文中移除 survey 块（解析问卷仍用原文）。 */
export function stripSurveyForDisplay(text: string): string {
  if (!text) {
    return '';
  }
  if (!text.includes(SURVEY_START) && !CONFIRMATION_HEADER_RE.test(text)) {
    return text;
  }
  const cleaned = stripConfirmationMarkdown(text.replace(SURVEY_BLOCK_RE, ''));
  return cleaned.trim();
}

/** 是否含结构化 survey 块（用于展示折叠提示）。 */
export function hasSurveyBlock(text: string): boolean {
  return text.includes(SURVEY_START);
}
