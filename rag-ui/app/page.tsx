"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  useChatRuntime,
  AssistantChatTransport,
} from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { useThreadPersistence } from "@/lib/hooks/use-thread-persistence";
import { useAutoThreadTitle } from "@/lib/hooks/use-auto-thread-title";
import { useState, useEffect } from "react";
import { DisclaimerDialog } from "@/components/disclaimer-dialog";

export default function Home() {
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({
      api: "/api/chat",
    }),
  });

  const [currentThreadTitle, setCurrentThreadTitle] = useState<string>("New Chat");

  // Initialize thread persistence with localStorage
  useThreadPersistence(runtime);

  // Initialize auto-naming for new chats
  useAutoThreadTitle(runtime);

  useEffect(() => {
    if (!runtime) return;

    const unsubscribe = runtime.threadList.subscribe(() => {
      try {
        const threadId = runtime.thread.getState().threadId;
        const itemRuntime = runtime.threadList.getItemById(threadId);
        const itemState = itemRuntime.getState();
        console.log("[Home] ThreadList changed, new title:", itemState.title);
        setCurrentThreadTitle(itemState.title || "New Chat");
      } catch (error) {
        // Thread may have been deleted, ignore the error
        console.log("[Home] Thread no longer exists (likely deleted)");
        setCurrentThreadTitle("New Chat");
      }
    });

    const threadUnsubscribe = runtime.thread.subscribe(() => {
      try {
        const threadId = runtime.thread.getState().threadId;
        const itemRuntime = runtime.threadList.getItemById(threadId);
        const itemState = itemRuntime.getState();
        console.log("[Home] Thread changed, new title:", itemState.title);
        setCurrentThreadTitle(itemState.title || "New Chat");
      } catch (error) {
        // Thread may have been deleted, ignore the error
        console.log("[Home] Thread no longer exists (likely deleted)");
        setCurrentThreadTitle("New Chat");
      }
    });

    return () => {
      unsubscribe?.();
      threadUnsubscribe?.();
    };
  }, [runtime]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <DisclaimerDialog />
      <SidebarProvider>
        <div className="flex h-dvh w-full pr-0.5">
          <ThreadListSidebar />
          <SidebarInset>
            <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
              <SidebarTrigger />
              <Separator orientation="vertical" className="mr-2 h-4" />
              <Breadcrumb>
                <BreadcrumbList>
                  <BreadcrumbItem className="hidden md:block">
                    <BreadcrumbLink
                      href="https://open.fda.gov/"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Medication Assistant
                    </BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator className="hidden md:block" />
                  <BreadcrumbItem>
                    <BreadcrumbPage>{currentThreadTitle}</BreadcrumbPage>
                  </BreadcrumbItem>
                </BreadcrumbList>
              </Breadcrumb>
            </header>
            <div className="flex-1 overflow-hidden">
              <Thread />
            </div>
          </SidebarInset>
        </div>
      </SidebarProvider>
    </AssistantRuntimeProvider>
  );
}
