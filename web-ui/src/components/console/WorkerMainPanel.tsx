import type { ChatMessage } from './ChatThread';
import ChatThread from './ChatThread';

const STATUS_LABEL: Record<string, string> = {
  done: '已完成',
  running: '执行中',
  waiting: '等待',
  failed: '失败',
  pending: '待执行',
  skipped: '已跳过',
};

interface Props {
  planTitle: string;
  taskId: string;
  taskTitle: string;
  taskStatus: string;
  messages: ChatMessage[];
  traceLines: string;
  busy: boolean;
  onBack: () => void;
}

export default function WorkerMainPanel({
  planTitle,
  taskId,
  taskTitle,
  taskStatus,
  messages,
  traceLines,
  busy,
  onBack,
}: Props) {
  const statusClass = `badge status-${taskStatus}`;
  const statusText = STATUS_LABEL[taskStatus] || taskStatus;

  return (
    <div className="cursor-worker-main">
      <div className="cursor-worker-main-toolbar">
        <button type="button" className="cursor-btn-ghost cursor-worker-back" onClick={onBack}>
          ← 返回 Plan
        </button>
        <div className="cursor-worker-main-heading">
          <span className="cursor-worker-main-plan">{planTitle}</span>
          <h2 className="cursor-worker-main-title">
            Work {taskId}
            {taskTitle && taskTitle !== taskId ? ` · ${taskTitle}` : ''}
          </h2>
        </div>
        <span className={statusClass}>{statusText}</span>
        {busy && taskStatus === 'running' && (
          <span className="badge badge-running">刷新中…</span>
        )}
      </div>

      <div className="cursor-worker-main-body">
        <ChatThread
          messages={messages}
          liveTraceText={traceLines}
          liveTraceSteps={[]}
          liveThinkingText=""
          streamText=""
          busy={busy && taskStatus === 'running'}
          traceMode="steps"
        />
        {messages.length === 0 && !traceLines.trim() && (
          <div className="cursor-worker-main-empty">
            {taskStatus === 'pending'
              ? '任务尚未开始，等待 Plan 调度…'
              : taskStatus === 'running'
                ? 'Worker 执行中，过程将自动刷新…'
                : '暂无执行记录'}
          </div>
        )}
      </div>
    </div>
  );
}
