interface Props {
  edge: 'left' | 'right';
  active?: boolean;
  title?: string;
  onMouseDown: (e: React.MouseEvent) => void;
}

export default function PanelResizeHandle({ edge, active = false, title, onMouseDown }: Props) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={title || '调整面板宽度'}
      title={title || '拖动调整宽度'}
      className={`cursor-resize-handle cursor-resize-handle--${edge}${active ? ' is-active' : ''}`}
      onMouseDown={onMouseDown}
    />
  );
}
