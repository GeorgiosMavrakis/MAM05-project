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
import { useAutoThreadTitle } from "@/lib/hooks/use-auto-thread-title";
import { useState, useEffect } from "react";

export const Assistant = () => {
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({
      api: "/api/chat",
    }),
  });

  const [currentThreadTitle, setCurrentThreadTitle] = useState<string>("New Chat");

  useAutoThreadTitle(runtime);

  useEffect(() => {
    if (!runtime) return;

    const unsubscribe = runtime.threadList.subscribe(() => {
      const threadId = runtime.thread.getState().threadId;
      const itemRuntime = runtime.threadList.getItemById(threadId);
      const itemState = itemRuntime.getState();
      console.log("[Assistant] ThreadList changed, new title:", itemState.title);
      setCurrentThreadTitle(itemState.title || "New Chat");
    });

    const threadUnsubscribe = runtime.thread.subscribe(() => {
      const threadId = runtime.thread.getState().threadId;
      const itemRuntime = runtime.threadList.getItemById(threadId);
      const itemState = itemRuntime.getState();
      console.log("[Assistant] Thread changed, new title:", itemState.title);
      setCurrentThreadTitle(itemState.title || "New Chat");
    });

    return () => {
      unsubscribe?.();
      threadUnsubscribe?.();
    };
  }, [runtime]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
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
                      href="https://www.assistant-ui.com/docs/getting-started"
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
};
