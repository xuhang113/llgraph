import type { Dispatch, MutableRefObject, SetStateAction } from 'react';

type StreamRefs = {
  runningThreads: MutableRefObject<Set<string>>;
  streamAbort: MutableRefObject<Map<string, AbortController>>;
  streamLastEventAt?: MutableRefObject<Map<string, number>>;
};

/** 释放某 thread 的本地流式占用（running / abort / lastEvent），可选同步 busy。 */
export function releaseStreamState(
  threadId: string,
  refs: StreamRefs,
  setBusy?: Dispatch<SetStateAction<boolean>>,
): void {
  if (!threadId) {
    return;
  }
  refs.runningThreads.current.delete(threadId);
  refs.streamAbort.current.delete(threadId);
  refs.streamLastEventAt?.current.delete(threadId);
  if (setBusy) {
    setBusy(false);
  }
}
