export const SIDEBAR_WIDTH_DEFAULT = 260;
export const SIDEBAR_WIDTH_MIN = 200;
export const SIDEBAR_WIDTH_MAX = 420;
export const RIGHT_PANEL_WIDTH_DEFAULT = 320;
export const RIGHT_PANEL_WIDTH_MIN = 260;
export const RIGHT_PANEL_WIDTH_MAX = 560;

/** SSE 超过此时间无消息则强制重连 EventSource。 */
export const SSE_STALE_MS = 12_000;
/** SSE 活跃时跳过 last-trace 轮询，避免用落盘快照覆盖实时 panel。 */
export const SSE_TRACE_POLL_SKIP_MS = 5_000;
/** POST /chat SSE 在此时间内仍有事件时，才屏蔽 Session 长连接的重复 trace。 */
export const POST_STREAM_ACTIVE_MS = 12_000;
/** 仅这些 SSE 类型视为「有新 trace 内容」，trace_activity 心跳不计入（避免阻塞 lastTrace 轮询）。 */
export const SSE_TRACE_CONTENT_TYPES = new Set([
  'trace_line',
  'trace_step',
  'turn_start',
  'turn_done',
  'thinking_delta',
  'stream_delta',
  'stream_end',
  'error',
  'end',
]);

export const LAST_SESSION_THREAD_KEY = 'llgraph.lastSessionThread';
