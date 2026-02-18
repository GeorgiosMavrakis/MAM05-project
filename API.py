from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import database, embedder
from fastapi.middleware.cors import CORSMiddleware
import llm_client
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
    return {
        "question": q,
        "answer": llm_client.ask_rag(q)
    }

@app.post("/chat")
async def chat(message: MessageRequest):
    """
    Endpoint for assistant-ui framework.
    Expects: {"content": "user question"}
    Returns streaming response compatible with assistant-ui
    """
    question = message.content

    async def generate():
        # Stream the response back to UI
        response_text = llm_client.ask_rag(question)
        yield json.dumps({"type": "text", "content": response_text}) + "\n"
        yield json.dumps({"type": "end"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(
        "API:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
