import type { SlashCatalogItem } from '../../api/client';

interface Props {
  items: SlashCatalogItem[];
  activeIndex: number;
  onSelect: (item: SlashCatalogItem) => void;
  onHover: (index: number) => void;
}

function badgeTone(category: string): string {
  if (category === 'Skills') {
    return 'skill';
  }
  if (category === 'Commands') {
    return 'command';
  }
  return 'meta';
}

/** 斜杠命令补全下拉（对齐终端 prompt_toolkit 样式）。 */
export default function SlashCompletionMenu({
  items,
  activeIndex,
  onSelect,
  onHover,
}: Props) {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className="cursor-slash-menu" role="listbox" aria-label="斜杠命令补全">
      {items.map((item, index) => (
        <button
          key={`${item.category}-${item.name}`}
          type="button"
          role="option"
          aria-selected={index === activeIndex}
          className={`cursor-slash-item${index === activeIndex ? ' is-active' : ''}`}
          onMouseDown={(e) => {
            e.preventDefault();
            onSelect(item);
          }}
          onMouseEnter={() => onHover(index)}
        >
          <span className="cursor-slash-item-main">
            <code className="cursor-slash-cmd">/{item.name}</code>
            <span className={`cursor-slash-badge cursor-slash-badge--${badgeTone(item.category)}`}>
              {item.badge}
            </span>
          </span>
          <span className="cursor-slash-desc">{item.description}</span>
        </button>
      ))}
    </div>
  );
}
