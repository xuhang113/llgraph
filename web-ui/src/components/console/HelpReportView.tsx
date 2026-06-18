import { parseHelpReport } from '../../utils/helpReport';
import MarkdownView from './MarkdownView';

interface Props {
  content: string;
}

/** 终端风格 /help 报告卡片。 */
export default function HelpReportView({ content }: Props) {
  const report = parseHelpReport(content);
  if (!report) {
    return <MarkdownView content={content} />;
  }

  return (
    <div className="cursor-help-report">
      <header className="cursor-help-report-header">
        <h3 className="cursor-help-report-title">{report.title}</h3>
        <p className="cursor-help-report-sub">会话内命令 · 不会发给模型</p>
      </header>

      <div className="cursor-help-report-body">
        {report.sections.map((section) => (
          <section key={section.title} className="cursor-help-section">
            <h4 className="cursor-help-section-title">{section.title}</h4>
            {section.title === '当前会话' ? (
              <div className="cursor-help-session">
                {section.lines.map((line) => (
                  <p key={line} className="cursor-help-session-line">
                    {line.split('|').map((part) => (
                      <span key={part.trim()} className="cursor-help-pill">
                        {part.trim()}
                      </span>
                    ))}
                  </p>
                ))}
              </div>
            ) : (
              <>
                {section.commands.map((row) => (
                  <div key={`${section.title}-${row.command}`} className="cursor-help-cmd-row">
                    <code className="cursor-help-cmd">{row.command}</code>
                    <span className="cursor-help-cmd-desc">{row.description}</span>
                  </div>
                ))}
                {section.lines.map((line) => (
                  <p
                    key={`${section.title}-${line}`}
                    className={`cursor-help-plain${line.includes('Ctrl') || line.includes('Alt') ? ' cursor-help-shortcut' : ''}`}
                  >
                    {line}
                  </p>
                ))}
              </>
            )}
          </section>
        ))}
      </div>

      {report.footerLines.length > 0 && (
        <footer className="cursor-help-footer">
          {report.footerLines.map((line) => (
            <p key={line}>{line}</p>
          ))}
        </footer>
      )}
    </div>
  );
}
