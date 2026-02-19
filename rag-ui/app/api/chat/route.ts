import { type UIMessage } from "ai";

// Get API endpoint from environment or use default
const API_ENDPOINT = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_TIMEOUT = parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT || "60", 10) * 1000;

export async function POST(req: Request) {
  try {
    const body = await req.json();
    console.log("[RAG Route] Received body:", JSON.stringify(body).substring(0, 500));

    // Handle both formats: direct content field or messages array
    let messageContent: string | null = null;

    // Check if body has direct "content" field (from streaming request)
    if (body.content) {
      messageContent = body.content;
    }
    // Check if body has "messages" array (from assistant-ui framework)
    else if (body.messages && Array.isArray(body.messages) && body.messages.length > 0) {
      const userMessage = body.messages[body.messages.length - 1];

      // Handle assistant-ui format: messages[].parts[].text
      if (userMessage.parts && Array.isArray(userMessage.parts)) {
        messageContent = userMessage.parts
          .map((part: any) => part.text || "")
          .join("");
      }
      // Handle AI SDK format: messages[].content
      else if (userMessage.content) {
        messageContent = typeof userMessage.content === "string"
          ? userMessage.content
          : userMessage.content.map((c: any) => c.text || "").join("");
      }
    }

    if (!messageContent) {
      console.error("[RAG Route] No message content found. Body:", JSON.stringify(body));
      return new Response("Empty message content", { status: 400 });
    }

    console.log(`[RAG Route] Processing message: ${messageContent.substring(0, 100)}`);

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
          content: messageContent,
        }),
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!response.ok) {
        console.error(`[RAG Client] API error: ${response.status} ${response.statusText}`);
        return new Response(
          `0:"Error: API request failed with status ${response.status}"\n`,
          {
            headers: {
              "Content-Type": "text/event-stream; charset=utf-8",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
            },
          }
        );
      }

      if (!response.body) {
        throw new Error("No response stream from API");
      }

      // Transform the NDJSON response to AI SDK format
      const transformStream = new TransformStream({
        transform(chunk: Uint8Array, controller) {
          const text = new TextDecoder().decode(chunk);
          const lines = text.split("\n");

          for (const line of lines) {
            if (!line.trim()) continue;

            try {
              const data = JSON.parse(line);
              console.log(`[RAG Transform] Processing:`, JSON.stringify(data).substring(0, 100));

              if (data.type === "text" && data.content) {
                // Escape the content for JSON string in AI SDK format
                const escaped = data.content
                  .replace(/\\/g, "\\\\")
                  .replace(/"/g, '\\"')
                  .replace(/\n/g, "\\n");
                const streamLine = `0:"${escaped}"\n`;
                console.log(`[RAG Transform] Sending: ${streamLine.substring(0, 100)}`);
                controller.enqueue(new TextEncoder().encode(streamLine));
              } else if (data.type === "end") {
                console.log("[RAG Transform] Received end signal from backend");
              } else if (data.type === "error") {
                // Treat errors as text but ensure they have error marker
                console.log(`[RAG Transform] Backend error: ${data.content}`);
                const errorMsg = data.content.startsWith("❌") ? data.content : `❌ ${data.content}`;
                const escaped = errorMsg.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
                const streamLine = `0:"${escaped}"\n`;
                console.log(`[RAG Transform] Sending error as text: ${streamLine.substring(0, 100)}`);
                controller.enqueue(new TextEncoder().encode(streamLine));
              }
            } catch (parseError) {
              console.warn(`[RAG Transform] Failed to parse JSON: ${line.substring(0, 100)}`);
            }
          }
        },
      });

      console.log("[RAG Route] Piping response through transform stream");
      const transformedStream = response.body.pipeThrough(transformStream);

      return new Response(transformedStream, {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } catch (streamError) {
      console.error("[RAG Route] Streaming error:", streamError);
      const errorMessage = streamError instanceof Error ? streamError.message : "Unknown error occurred";
      const escaped = errorMessage.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
      return new Response(`0:"❌ Connection Error: ${escaped}"\n`, {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }
  } catch (error) {
    console.error("[RAG Route] Request error:", error);
    const errorMsg = error instanceof Error ? error.message : "Internal server error";
    const escaped = errorMsg.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
    return new Response(`0:"❌ Server Error: ${escaped}"\n`, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }
}


