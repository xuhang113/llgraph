import type { ReactNode } from 'react';
import CopyButton from './CopyButton';

interface Props {
  code: string;
  children: ReactNode;
}

export default function MarkdownCodeBlock({ code, children }: Props) {
  return (
    <div className="markdown-code-block">
      <CopyButton text={code} title="复制代码" className="markdown-code-block-copy" iconOnly />
      <pre>{children}</pre>
    </div>
  );
}
