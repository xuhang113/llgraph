import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ReactNode } from 'react';
import { contentToText } from '../../utils/format';

interface Props {
  content: string;
  className?: string;
}

const markdownComponents = {
  table: ({ children }: { children?: ReactNode }) => (
    <div className="markdown-table-wrap">
      <table>{children}</table>
    </div>
  ),
};

export default function MarkdownView({ content, className = '' }: Props) {
  if (!content.trim()) {
    return null;
  }
  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

export function MessageContent({ content }: { content: unknown }) {
  const text = contentToText(content);
  return <MarkdownView content={text} />;
}
