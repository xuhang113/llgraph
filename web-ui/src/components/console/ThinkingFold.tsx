interface Props {
  /** 各轮 ReAct 推理摘要（按时间顺序） */
  segments: string[];
  /** 流式 thinking 全文（进行中） */
  liveText?: string;
  live?: boolean;
}

export default function ThinkingFold({ segments, liveText = '', live = false }: Props) {
  const rounds = segments.map((s) => s.trim()).filter(Boolean);
  const streaming = liveText.trim();
  const body = streaming || rounds.join('\n\n');
  const charCount = body.length;
  const roundCount = rounds.length;

  if (!body && !live) {
    return null;
  }

  const meta =
    charCount > 0
      ? roundCount > 1
        ? `${roundCount} 轮 · ${charCount} 字`
        : `${charCount} 字`
      : '';

  return (
    <details className="cursor-thinking-fold" open={live}>
      <summary className="cursor-thinking-fold-summary">
        <span className="cursor-thinking-fold-icon" aria-hidden>
          ◎
        </span>
        <span className="cursor-thinking-fold-label">{live ? '思考中' : '思考过程'}</span>
        {meta && <span className="cursor-thinking-fold-meta">{meta}</span>}
        {!live && <span className="cursor-thinking-fold-hint">点击展开</span>}
      </summary>
      <div className="cursor-thinking-fold-body">
        {streaming ? (
          <div className="cursor-thinking-fold-segment">{streaming}</div>
        ) : (
          rounds.map((seg, index) => (
            <div key={index} className="cursor-thinking-fold-segment">
              {roundCount > 1 && (
                <div className="cursor-thinking-fold-round">第 {index + 1} 轮</div>
              )}
              {seg}
            </div>
          ))
        )}
      </div>
    </details>
  );
}
