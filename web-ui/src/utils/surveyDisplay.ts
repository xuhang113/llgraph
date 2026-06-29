/** 与后端 survey_prompt.py 对齐：对话区仅隐藏 <<<llgraph-survey>>> 块 */

const SURVEY_START = '<<<llgraph-survey>>>';
const SURVEY_END = '<<<end-survey>>>';
const SURVEY_BLOCK_RE = new RegExp(
  `${escapeRegExp(SURVEY_START)}[\\s\\S]*?(?:${escapeRegExp(SURVEY_END)}|$)`,
  'g',
);

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** 从展示用正文中移除 survey 块（解析问卷仍用原文）。 */
export function stripSurveyForDisplay(text: string): string {
  if (!text || !text.includes(SURVEY_START)) {
    return text;
  }
  return text.replace(SURVEY_BLOCK_RE, '').trim();
}

/** 是否含结构化 survey 块（用于展示折叠提示）。 */
export function hasSurveyBlock(text: string): boolean {
  return text.includes(SURVEY_START);
}
