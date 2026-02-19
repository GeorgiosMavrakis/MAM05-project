"use client";

import { useEffect, useRef } from "react";
import type { AssistantRuntime } from "@assistant-ui/react";
import { generateTitleFromMessage } from "@/lib/utils/title-generator";

/**
 * Hook to automatically generate and set thread titles based on the first user message
 */
export function useAutoThreadTitle(runtime: AssistantRuntime | null | undefined) {
  const processedThreadsRef = useRef<Set<string>>(new Set());
  const lastMessageCountRef = useRef<{ [threadId: string]: number }>({});

  useEffect(() => {
    if (!runtime) return;

    console.log("[AutoThreadTitle] Hook initialized");

    // Function to attempt title generation for a thread
    const trySetTitleForThread = (threadId: string) => {
      const itemRuntime = runtime.threadList.getItemById(threadId);
      const itemState = itemRuntime.getState();

      console.log("[AutoThreadTitle] Checking thread:", {
        threadId,
        hasTitle: !!itemState.title,
        alreadyProcessed: processedThreadsRef.current.has(threadId),
      });

      // Skip if title already set
      if (itemState.title) {
        processedThreadsRef.current.add(threadId);
        return;
      }

      // If we've already processed this thread, skip
      if (processedThreadsRef.current.has(threadId)) {
        return;
      }

      // Get the thread to check for messages
      const threadRuntime = runtime.threadList.getById(threadId);
      if (!threadRuntime) {
        console.log("[AutoThreadTitle] Cannot get thread runtime for:", threadId);
        return;
      }

      const threadState = threadRuntime.getState();
      const messages = threadState.messages;

      console.log("[AutoThreadTitle] Thread state check:", {
        threadId,
        messageCount: messages.length,
        lastCount: lastMessageCountRef.current[threadId],
      });

      // Look for the first user message in the thread
      const firstUserMessage = messages.find((msg) => msg.role === "user");

      if (firstUserMessage) {
        console.log("[AutoThreadTitle] Found first user message");

        // Extract text content - handle multiple formats
        let messageText = "";

        if (typeof firstUserMessage.content === "string") {
          messageText = firstUserMessage.content;
        } else if (Array.isArray(firstUserMessage.content)) {
          const textParts = firstUserMessage.content.filter(
            (part: any) => part.type === "text"
          );
          messageText = textParts.map((part: any) => part.text).join(" ");
        } else if (Array.isArray((firstUserMessage as any).parts)) {
          // Handle 'parts' array format
          const textParts = (firstUserMessage as any).parts.filter(
            (part: any) => part.type === "text" || typeof part === "string"
          );
          messageText = textParts.map((part: any) => part.text || part).join(" ");
        }

        console.log("[AutoThreadTitle] Message text:", messageText.substring(0, 100));

        if (messageText && messageText.trim()) {
          const generatedTitle = generateTitleFromMessage(messageText);

          console.log("[AutoThreadTitle] Generated title:", generatedTitle);

          // Only set title if it's different from the fallback
          if (generatedTitle && generatedTitle !== "New Chat") {
            console.log("[AutoThreadTitle] Setting title:", generatedTitle);
            itemRuntime
              .rename(generatedTitle)
              .then(() => {
                console.log("[AutoThreadTitle] Successfully renamed thread");
              })
              .catch((error) =>
                console.error("[AutoThreadTitle] Failed to rename thread:", error)
              );
            processedThreadsRef.current.add(threadId);
          }
        }
      } else {
        console.log("[AutoThreadTitle] No first user message found yet");
      }
    };

    // Subscribe to thread changes to detect new messages
    const threadUnsubscribe = runtime.thread.subscribe(() => {
      const threadId = runtime.thread.getState().threadId;
      const threadState = runtime.thread.getState();
      lastMessageCountRef.current[threadId] = threadState.messages.length;
      console.log("[AutoThreadTitle] Thread changed event");
      trySetTitleForThread(threadId);
    });

    // Also subscribe to threadList changes to detect new threads
    const threadListUnsubscribe = runtime.threadList.subscribe(() => {
      const currentThreadId = runtime.thread.getState().threadId;
      console.log("[AutoThreadTitle] ThreadList changed event");
      trySetTitleForThread(currentThreadId);
    });

    return () => {
      threadUnsubscribe?.();
      threadListUnsubscribe?.();
    };
  }, [runtime]);
}

