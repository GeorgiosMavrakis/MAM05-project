from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import llm_client
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel

app = FastAPI()

class MessageRequest(BaseModel):
    content: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
def ask(q: str):
    """
    Synchronous endpoint for simple Q&A.
    Returns full answer with citations.
    """
    answer = llm_client.ask_rag(q)
    return {
        "question": q,
        "answer": answer
    }

@app.post("/chat")
async def chat(message: MessageRequest):
    """
    Endpoint for assistant-ui framework.
    Expects: {"content": "user question"}
    Returns streaming response compatible with assistant-ui

    Streams JSON objects:
    - {"type": "text", "content": "..."}
    - {"type": "end"}
    """
    question = message.content

    async def generate():
        try:
            # Stream response from RAG pipeline
            for chunk in llm_client.ask_rag_streaming(question):
                yield json.dumps({"type": "text", "content": chunk}) + "\n"

            # Send end signal
            yield json.dumps({"type": "end"}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "Medical RAG API"}

if __name__ == "__main__":
    uvicorn.run(
        "API:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
