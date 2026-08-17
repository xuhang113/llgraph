import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Children, isValidElement, type ReactNode } from 'react';
import { contentToText } from '../../utils/format';
import { normalizeMarkdownForRender } from '../../utils/markdownNormalize';
import MermaidChart from './MermaidChart';
import MarkdownCodeBlock from './MarkdownCodeBlock';

interface Props {
  content: string;
  className?: string;
  /** 流式输出：容忍尾部不完整块，并做增量友好归一化 */
  streaming?: boolean;
}

function extractMermaidSource(children: ReactNode): string | null {
  const child = Children.only(children);
  if (!isValidElement(child)) {
    return null;
  }
  const props = child.props as { className?: string; children?: ReactNode };
  const className = props.className || '';
  if (!className.includes('language-mermaid')) {
    return null;
  }
  return String(props.children ?? '').replace(/\n$/, '');
}

function extractCodeText(children: ReactNode): string {
  const child = Children.only(children);
  if (!isValidElement(child)) {
    return String(children ?? '').replace(/\n$/, '');
  }
  const props = child.props as { children?: ReactNode };
  return String(props.children ?? '').replace(/\n$/, '');
}

const markdownComponents = {
  pre: ({ children }: { children?: ReactNode }) => {
    const source = children ? extractMermaidSource(children) : null;
    if (source !== null) {
      return <MermaidChart code={source} />;
    }
    const code = children ? extractCodeText(children) : '';
    return <MarkdownCodeBlock code={code}>{children}</MarkdownCodeBlock>;
  },
  table: ({ children }: { children?: ReactNode }) => (
    <div className="markdown-table-wrap">
      <table>{children}</table>
    </div>
  ),
};

export default function MarkdownView({ content, className = '', streaming = false }: Props) {
  if (!content.trim()) {
    return null;
  }
  const normalized = normalizeMarkdownForRender(content, { streaming });
  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {normalized}
      </ReactMarkdown>
    </div>
  );
}

export function MessageContent({ content }: { content: unknown }) {
  const text = contentToText(content);
  return <MarkdownView content={text} />;
}
