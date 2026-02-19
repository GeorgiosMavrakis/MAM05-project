import type { UIMessage } from "ai";

// Get API endpoint from environment or use default
const API_ENDPOINT = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_TIMEOUT =
  parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT || "60", 10) * 1000;

export async function POST(req: Request) {
  try {
    const body = await req.json();
    console.log(
      "[RAG Route] Full request body:",
      JSON.stringify(body).substring(0, 500),
    );

    const { messages }: { messages: UIMessage[] } = body;
    console.log("[RAG Route] Received messages, count:", messages?.length);

    if (!messages || messages.length === 0) {
      console.error("[RAG Route] No messages provided");
      return new Response("Empty messages", { status: 400 });
    }

    // Get the last user message
    const lastMessage = messages[messages.length - 1];
    console.log(
      "[RAG Route] Last message:",
      JSON.stringify(lastMessage).substring(0, 300),
    );

    let messageContent: string = "";

    // Handle different message formats - assistant-ui uses 'parts' array
    if (typeof lastMessage.content === "string") {
      messageContent = lastMessage.content;
      console.log(
        "[RAG Route] Content is string:",
        messageContent.substring(0, 50),
      );
    } else if (Array.isArray(lastMessage.content)) {
      console.log(
        "[RAG Route] Content is array, parts count:",
        lastMessage.content.length,
      );
      messageContent = lastMessage.content
        .map((part: any) => {
          console.log(
            "[RAG Route] Part:",
            JSON.stringify(part).substring(0, 100),
          );
          return part.text || "";
        })
        .join("");
    } else if (Array.isArray((lastMessage as any).parts)) {
      // assistant-ui format uses 'parts' instead of 'content'
      console.log(
        "[RAG Route] Message uses 'parts' array, count:",
        (lastMessage as any).parts.length,
      );
      messageContent = (lastMessage as any).parts
        .map((part: any) => {
          console.log(
            "[RAG Route] Part:",
            JSON.stringify(part).substring(0, 100),
          );
          return part.text || "";
        })
        .join("");
    } else {
      console.log(
        "[RAG Route] Content type:",
        typeof lastMessage.content,
        "Value:",
        lastMessage.content,
      );
    }

    if (!messageContent) {
      console.error(
        "[RAG Route] No message content found. Message object:",
        JSON.stringify(lastMessage),
      );
      return new Response("Empty message content", { status: 400 });
    }

    console.log(
      `[RAG Route] Processing message: ${messageContent.substring(0, 100)}`,
    );

    try {
      const chatUrl = `${API_ENDPOINT}/chat`;
      console.log(`[RAG Client] Connecting to API: ${chatUrl}`);

      const backendResponse = await fetch(chatUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/x-ndjson",
        },
        body: JSON.stringify({
          content: messageContent,
        }),
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!backendResponse.ok) {
        console.error(
          `[RAG Client] API error: ${backendResponse.status} ${backendResponse.statusText}`,
        );
        throw new Error(`Backend API error: ${backendResponse.status}`);
      }

      if (!backendResponse.body) {
        throw new Error("No response stream from API");
      }

      // Transform the NDJSON stream to AI SDK format and return directly
      const transformedStream = transformNDJSONToAIStream(backendResponse.body);

      return new Response(transformedStream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } catch (streamError) {
      console.error("[RAG Route] Streaming error:", streamError);
      throw streamError;
    }
  } catch (error) {
    console.error("[RAG Route] Request error:", error);
    const errorMsg =
      error instanceof Error ? error.message : "Internal server error";
    return new Response(`Error: ${errorMsg}`, { status: 500 });
  }
}

function transformNDJSONToAIStream(
  ndjsonStream: ReadableStream<Uint8Array>,
): ReadableStream<Uint8Array> {
  let buffer = "";
  const messageId = `msg_${Date.now()}`;

  return new ReadableStream({
    async start(controller) {
      const reader = ndjsonStream.getReader();
      const decoder = new TextDecoder();
      let isClosed = false;
      let hasStarted = false;

      try {
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            // Process any remaining buffer
            if (buffer.trim() && !isClosed) {
              try {
                const data = JSON.parse(buffer);
                if (data.type === "text" && data.content) {
                  console.log(
                    `[AIStream] Final chunk: "${data.content.substring(0, 50)}"`,
                  );
                  if (!hasStarted) {
                    controller.enqueue(createTextStartEvent(messageId));
                    hasStarted = true;
                  }
                  controller.enqueue(
                    createTextDeltaEvent(messageId, data.content),
                  );
                }
              } catch (_e) {
                console.warn(`[AIStream] Failed to parse final buffer`);
              }
            }

            // Send text-end event
            if (!isClosed) {
              console.log(`[AIStream] Sending text-end event`);
              if (!hasStarted) {
                controller.enqueue(createTextStartEvent(messageId));
              }
              controller.enqueue(createTextEndEvent(messageId));

              try {
                controller.close();
              } catch (_closeError) {
                console.log(`[AIStream] Controller already closed on done`);
              }
              isClosed = true;
            }
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.trim() || isClosed) continue;

            try {
              const data = JSON.parse(line);
              console.log(
                `[AIStream] Processing:`,
                JSON.stringify(data).substring(0, 100),
              );

              if (data.type === "text" && data.content) {
                console.log(
                  `[AIStream] Chunk: "${data.content.substring(0, 50)}"`,
                );
                if (!hasStarted) {
                  controller.enqueue(createTextStartEvent(messageId));
                  hasStarted = true;
                }
                controller.enqueue(
                  createTextDeltaEvent(messageId, data.content),
                );
              } else if (data.type === "error") {
                const errorMsg = data.content.startsWith("❌")
                  ? data.content
                  : `❌ ${data.content}`;
                console.log(`[AIStream] Error: ${errorMsg}`);
                if (!hasStarted) {
                  controller.enqueue(createTextStartEvent(messageId));
                  hasStarted = true;
                }
                controller.enqueue(createErrorEvent(errorMsg));
              } else if (data.type === "end") {
                console.log(`[AIStream] End signal received`);
                if (!isClosed) {
                  buffer = "";
                  try {
                    controller.close();
                  } catch (_closeError) {
                    console.log(`[AIStream] Controller already closed`);
                  }
                  isClosed = true;
                }
              }
            } catch (_parseError) {
              console.warn(
                `[AIStream] Failed to parse JSON: ${line.substring(0, 100)}`,
              );
            }
          }

          if (isClosed) break;
        }
      } catch (error) {
        console.error("[AIStream] Error reading stream:", error);
        if (!isClosed) {
          try {
            controller.error(error);
          } catch (_e) {
            console.log("[AIStream] Error on controller.error");
          }
        }
      } finally {
        reader.releaseLock();
      }
    },
  });
}

function createTextStartEvent(id: string): Uint8Array {
  const event = {
    type: "text-start",
    id: id,
  };
  return new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
}

function createTextDeltaEvent(id: string, delta: string): Uint8Array {
  const event = {
    type: "text-delta",
    id: id,
    delta: delta,
  };
  return new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
}

function createTextEndEvent(id: string): Uint8Array {
  const event = {
    type: "text-end",
    id: id,
  };
  return new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
}

function createErrorEvent(errorText: string): Uint8Array {
  const event = {
    type: "error",
    errorText: errorText,
  };
  return new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
}
