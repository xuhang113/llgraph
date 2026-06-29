import type { TraceStep } from '../../types/trace';
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
  taskReadonly?: boolean;
  messages: ChatMessage[];
  traceLines: string;
  liveTraceSteps?: TraceStep[];
  busy: boolean;
  onBack: () => void;
  onStop?: () => void;
  onRun?: () => void;
}

export default function WorkerMainPanel({
  planTitle,
  taskId,
  taskTitle,
  taskStatus,
  taskReadonly = false,
  messages,
  traceLines,
  liveTraceSteps = [],
  busy,
  onBack,
  onStop,
  onRun,
}: Props) {
  const statusClass = `badge status-${taskStatus}`;
  const statusText = STATUS_LABEL[taskStatus] || taskStatus;
  const modeLabel = taskReadonly ? '只读' : '可写';

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
        <span className={`badge${taskReadonly ? '' : ' badge-running'}`}>{modeLabel}</span>
        <span className={statusClass}>{statusText}</span>
        {busy && taskStatus === 'running' && (
          <span className="badge badge-running">刷新中…</span>
        )}
        {taskStatus === 'running' && onStop && (
          <button type="button" className="cursor-btn-danger cursor-btn-sm" onClick={onStop}>
            停止
          </button>
        )}
        {(taskStatus === 'pending' || taskStatus === 'failed' || taskStatus === 'skipped') && onRun && (
          <button type="button" className="cursor-btn-ghost cursor-btn-sm" onClick={onRun}>
            执行
          </button>
        )}
      </div>

      <div className="cursor-worker-main-body">
        <ChatThread
          messages={messages}
          liveTraceText={traceLines}
          liveTraceSteps={liveTraceSteps}
          streamText=""
          busy={busy && taskStatus === 'running'}
          traceMode="steps"
        />
        {messages.length === 0 && !traceLines.trim() && liveTraceSteps.length === 0 && (
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
