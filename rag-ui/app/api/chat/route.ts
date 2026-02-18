import { type UIMessage } from "ai";

// Get API endpoint from environment or use default
const API_ENDPOINT = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_TIMEOUT = parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT || "60", 10) * 1000;

export async function POST(req: Request) {
  try {
    const { messages }: { messages: UIMessage[] } = await req.json();

    if (!messages || messages.length === 0) {
      return new Response("No messages provided", { status: 400 });
    }

    const userMessage = messages[messages.length - 1];

    if (!userMessage || !userMessage.content) {
      return new Response("Empty message content", { status: 400 });
    }

    async function* streamFromAPI() {
      let controller: AbortController | null = new AbortController();
      const timeoutId = setTimeout(() => controller?.abort(), API_TIMEOUT);

      try {
        const chatUrl = `${API_ENDPOINT}/chat`;
        console.log(`[RAG Client] Connecting to API: ${chatUrl}`);

        const response = await fetch(chatUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/x-ndjson",
          },
          body: JSON.stringify({
            content: userMessage.content,
          }),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          console.error(`[RAG Client] API error: ${response.status} ${response.statusText}`);
          throw new Error(
            `API request failed: ${response.status} ${response.statusText}`
          );
        }

        if (!response.body) {
          throw new Error("No response stream from API");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        console.log("[RAG Client] Starting stream...");

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            console.log("[RAG Client] Stream ended");
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep incomplete line in buffer

          for (const line of lines) {
            if (!line.trim()) continue;

            try {
              const data = JSON.parse(line);

              // Handle different response types from API
              if (data.type === "text" && data.content) {
                console.log(`[RAG Client] Received text chunk: ${data.content.length} chars`);
                yield data.content;
              } else if (data.type === "end") {
                console.log("[RAG Client] Received end signal");
                // End signal received, continue to check for more data
              } else if (data.type === "error") {
                console.error(`[RAG Client] API error: ${data.content}`);
                yield `\n\n⚠️ Error from API: ${data.content}`;
              } else {
                console.warn(`[RAG Client] Unknown response type: ${data.type}`);
              }
            } catch (parseError) {
              console.warn(`[RAG Client] Failed to parse JSON: ${line.substring(0, 100)}`);
              // Skip invalid JSON lines
            }
          }
        }

        // Send any remaining buffered data
        if (buffer.trim()) {
          try {
            const data = JSON.parse(buffer);
            if (data.type === "text" && data.content) {
              yield data.content;
            }
          } catch {
            // Ignore parse errors on final buffer
          }
        }
      } catch (error) {
        clearTimeout(timeoutId);
        controller = null;

        if (error instanceof Error) {
          if (error.name === "AbortError") {
            console.error("[RAG Client] Request timeout");
            yield `\n\n⚠️ Request timeout (${API_TIMEOUT / 1000}s exceeded). Please try again.`;
          } else {
            console.error(`[RAG Client] Stream error: ${error.message}`);
            yield `\n\n⚠️ Connection error: ${error.message}. Make sure the API server is running at ${API_ENDPOINT}`;
          }
        } else {
          console.error("[RAG Client] Unknown error occurred");
          yield `\n\n⚠️ Unknown error occurred`;
        }
      }
    }

    // Convert async generator to ReadableStream
    const readable = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of streamFromAPI()) {
            controller.enqueue(new TextEncoder().encode(chunk));
          }
          controller.close();
        } catch (error) {
          console.error("[RAG Stream] Fatal error:", error);
          controller.error(error);
        }
      },
    });

    return new Response(readable, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "Transfer-Encoding": "chunked",
      },
    });
  } catch (error) {
    console.error("[RAG Route] Request error:", error);
    return new Response("Internal server error", { status: 500 });
  }
}
