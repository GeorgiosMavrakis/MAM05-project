"use client";

import { useEffect, useMemo, useRef } from "react";
import type { AssistantRuntime } from "@assistant-ui/react";

const DEFAULT_STORAGE_KEY = "assistant_ui_threads_v1";

type PersistedThread = {
  state: unknown;
  title?: string;
  archived?: boolean;
};

type PersistedThreads = {
  version: 1;
  currentThreadIndex: number;
  threads: PersistedThread[];
};

function isBrowser(): boolean {
  return (
    typeof window !== "undefined" && typeof window.localStorage !== "undefined"
  );
}

export function useThreadPersistence(
  runtime: AssistantRuntime | null | undefined,
  storageKey: string = DEFAULT_STORAGE_KEY,
) {
  const isRestoringRef = useRef(false);

  const persist = useMemo(() => {
    if (!runtime) return null;

    return () => {
      if (!isBrowser() || isRestoringRef.current) return;

      const threadList = runtime.threadList.getState();
      const activeThreadId = runtime.thread.getState().threadId;
      const threadIds = [
        ...threadList.threadIds,
        ...threadList.archivedThreadIds,
      ];

      const threads: PersistedThread[] = threadIds.map((threadId) => {
        const threadRuntime = runtime.threadList.getById(threadId);
        const itemRuntime = runtime.threadList.getItemById(threadId);
        const itemState = itemRuntime.getState();

        return {
          state: threadRuntime.exportExternalState(),
          title: itemState.title,
          archived: threadList.archivedThreadIds.includes(threadId),
        };
      });

      const currentThreadIndex = Math.max(0, threadIds.indexOf(activeThreadId));

      const payload: PersistedThreads = {
        version: 1,
        currentThreadIndex,
        threads,
      };

      try {
        window.localStorage.setItem(storageKey, JSON.stringify(payload));
      } catch (error) {
        console.error("[ThreadPersistence] Failed to persist threads", error);
      }
    };
  }, [runtime, storageKey]);

  useEffect(() => {
    if (!runtime || !isBrowser()) return;

    const load = async () => {
      if (isRestoringRef.current) return;

      const raw = window.localStorage.getItem(storageKey);
      if (!raw) return;

      let data: PersistedThreads | null = null;
      try {
        data = JSON.parse(raw) as PersistedThreads;
      } catch (error) {
        console.warn("[ThreadPersistence] Invalid persisted data", error);
        return;
      }

      if (!data || data.version !== 1 || data.threads.length === 0) return;

      isRestoringRef.current = true;

      const createdThreadIds: string[] = [];
      const applyThreadData = async (threadIndex: number, threadId: string) => {
        const threadData = data?.threads[threadIndex];
        if (!threadData) return;

        runtime.thread.importExternalState(threadData.state);

        if (threadData.title) {
          await runtime.threadList
            .getItemById(threadId)
            .rename(threadData.title);
        }
        if (threadData.archived) {
          await runtime.threadList.getItemById(threadId).archive();
        }
      };

      const currentMainId = runtime.thread.getState().threadId;
      createdThreadIds.push(currentMainId);
      await applyThreadData(0, currentMainId);

      for (let idx = 1; idx < data.threads.length; idx += 1) {
        await runtime.threadList.switchToNewThread();
        const newThreadId = runtime.thread.getState().threadId;
        createdThreadIds.push(newThreadId);
        await applyThreadData(idx, newThreadId);
      }

      const desiredIndex = Math.min(
        Math.max(0, data.currentThreadIndex),
        createdThreadIds.length - 1,
      );

      const targetThreadId = createdThreadIds[desiredIndex];
      if (
        targetThreadId &&
        targetThreadId !== runtime.thread.getState().threadId
      ) {
        await runtime.threadList.switchToThread(targetThreadId);
      }

      isRestoringRef.current = false;
      persist?.();
    };

    void load();
  }, [runtime, storageKey, persist]);

  useEffect(() => {
    if (!runtime || !persist) return;

    const unsubscribeThread = runtime.thread.subscribe(persist);
    const unsubscribeThreadList = runtime.threadList.subscribe(persist);

    return () => {
      unsubscribeThread?.();
      unsubscribeThreadList?.();
    };
  }, [runtime, persist]);
}
