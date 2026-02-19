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

// Global flag to prevent persisting during last thread deletion
let isDeletingLastThread = false;

export function setDeletingLastThread(value: boolean) {
  isDeletingLastThread = value;
  console.log("[ThreadPersistence] setDeletingLastThread:", value);
}

export function useThreadPersistence(
  runtime: AssistantRuntime | null | undefined,
  storageKey: string = DEFAULT_STORAGE_KEY,
) {
  const isRestoringRef = useRef(false);
  const lastThreadCountRef = useRef<number>(0);

  const persist = useMemo(() => {
    if (!runtime) return null;

    return () => {
      if (!isBrowser() || isRestoringRef.current) return;

      // Don't persist if we're in the middle of deleting the last thread
      if (isDeletingLastThread) {
        console.log("[ThreadPersistence] Skipping persist during last thread deletion");
        return;
      }

      const threadList = runtime.threadList.getState();
      const activeThreadId = runtime.thread.getState().threadId;
      const threadIds = [
        ...threadList.threadIds,
        ...threadList.archivedThreadIds,
      ];

      const currentThreadCount = threadIds.length;

      // If we went from having threads to having 0, clear storage
      if (lastThreadCountRef.current > 0 && currentThreadCount === 0) {
        try {
          window.localStorage.removeItem(storageKey);
          console.log("[ThreadPersistence] All threads deleted, cleared storage");
        } catch (error) {
          console.error("[ThreadPersistence] Failed to clear storage", error);
        }
        lastThreadCountRef.current = currentThreadCount;
        return;
      }

      // If only one thread exists and it's empty, don't persist
      if (currentThreadCount === 1) {
        const singleThreadId = threadIds[0];
        try {
          const threadRuntime = runtime.threadList.getById(singleThreadId);
          const threadState = threadRuntime.getState();

          if (threadState.messages.length === 0 && lastThreadCountRef.current >= 1) {
            try {
              window.localStorage.removeItem(storageKey);
              console.log("[ThreadPersistence] Last thread deleted, cleared storage");
            } catch (error) {
              console.error("[ThreadPersistence] Failed to clear storage", error);
            }
            lastThreadCountRef.current = 0;
            return;
          }
        } catch (error) {
          console.log("[ThreadPersistence] Could not get thread state", error);
        }
      }

      const threads: PersistedThread[] = threadIds.map((threadId) => {
        try {
          const threadRuntime = runtime.threadList.getById(threadId);
          const itemRuntime = runtime.threadList.getItemById(threadId);
          const itemState = itemRuntime.getState();

          return {
            state: threadRuntime.exportExternalState(),
            title: itemState.title,
            archived: threadList.archivedThreadIds.includes(threadId),
          };
        } catch (error) {
          console.log("[ThreadPersistence] Thread no longer available:", threadId);
          // Return a placeholder that will be filtered out
          return null;
        }
      }).filter((thread): thread is PersistedThread => thread !== null);

      // If no valid threads remain after filtering, clear storage
      if (threads.length === 0) {
        try {
          window.localStorage.removeItem(storageKey);
          console.log("[ThreadPersistence] No valid threads to persist, cleared storage");
        } catch (error) {
          console.error("[ThreadPersistence] Failed to clear storage", error);
        }
        lastThreadCountRef.current = 0;
        return;
      }

      const currentThreadIndex = Math.max(0, threadIds.indexOf(activeThreadId));

      const payload: PersistedThreads = {
        version: 1,
        currentThreadIndex,
        threads,
      };

      try {
        window.localStorage.setItem(storageKey, JSON.stringify(payload));
        lastThreadCountRef.current = currentThreadCount;
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
      if (!raw) {
        console.log("[ThreadPersistence] No saved data found");
        return;
      }

      let data: PersistedThreads | null = null;
      try {
        data = JSON.parse(raw) as PersistedThreads;
      } catch (error) {
        console.warn("[ThreadPersistence] Invalid persisted data", error);
        window.localStorage.removeItem(storageKey);
        return;
      }

      if (!data || data.version !== 1 || data.threads.length === 0) {
        console.log("[ThreadPersistence] No valid threads to restore");
        return;
      }

      // If we only have 1 thread and it has no messages, don't restore it
      if (data.threads.length === 1) {
        const singleThread = data.threads[0];
        const threadState = singleThread.state as any;

        if (threadState?.messages && Array.isArray(threadState.messages) && threadState.messages.length === 0) {
          console.log("[ThreadPersistence] Skipping restoration of single empty thread");
          window.localStorage.removeItem(storageKey);
          return;
        }
      }

      console.log("[ThreadPersistence] Restoring", data.threads.length, "thread(s) - FAST MODE");
      isRestoringRef.current = true;

      try {
        // FAST MODE: Only restore the most recent thread (the one user was viewing)
        // This prevents the freezing caused by creating multiple threads
        const targetIndex = Math.min(
          Math.max(0, data.currentThreadIndex),
          data.threads.length - 1,
        );
        const targetThread = data.threads[targetIndex];

        if (targetThread) {
          // Import the thread state into the current thread
          runtime.thread.importExternalState(targetThread.state);

          // Set the title
          const currentThreadId = runtime.thread.getState().threadId;
          if (targetThread.title) {
            await runtime.threadList.getItemById(currentThreadId).rename(targetThread.title);
          }

          console.log("[ThreadPersistence] Restored most recent thread");
        }

        // Initialize thread count
        lastThreadCountRef.current = data.threads.length;
        console.log("[ThreadPersistence] Restoration complete (fast mode)");
      } catch (error) {
        console.error("[ThreadPersistence] Error during restoration:", error);
        window.localStorage.removeItem(storageKey);
      } finally {
        isRestoringRef.current = false;
      }
    };

    void load();
  }, [runtime, storageKey]);

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
