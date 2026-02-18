from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from retriever import retrieve
from context_assembly import ContextAssembler
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    Workflow:
    1. User enters query in UI
    2. Retriever converts query to vectors and searches Qdrant
    3. Returns top N relevant chunks
    4. ContextAssembler formats chunks into prompt
    5. LLM generates patient-friendly answer
    6. Answer with citations returned to UI
    """
    try:
        logger.info(f"📝 Processing query: {q}")

        # STEP 1: Retrieve relevant chunks from Qdrant
        logger.info("🔍 Retrieving chunks from vector database...")
        retrieval_result = retrieve(q)
        text_chunks = retrieval_result.get("text_chunks", [])

        if not text_chunks:
            logger.warning("❌ No chunks found for query")
            return {
                "question": q,
                "answer": "No relevant information found in the database. Please try rewording your question.",
                "citations": [],
                "confidence": "low"
            }

        logger.info(f"✅ Retrieved {len(text_chunks)} chunks")

        # STEP 2: Assemble context from chunks
        logger.info("⚙️  Assembling context...")
        assembler = ContextAssembler()
        context = assembler.assemble_context(
            chunks=text_chunks,
            drug_brand="",
            drug_generic="",
            drug_class="",
            query=q
        )

        # STEP 3: Generate answer with LLM
        logger.info("🤖 Generating answer with LLM...")
        llm_answer = assembler.generate_answer_with_citations(
            context=context,
            query=q,
            chunks=text_chunks
        )

        logger.info(f"✅ Answer generated with confidence: {llm_answer.confidence}")

        return {
            "question": q,
            "answer": llm_answer.answer,
            "citations": llm_answer.citations,
            "confidence": llm_answer.confidence
        }

    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}", exc_info=True)
        return {
            "question": q,
            "answer": f"Error processing your question: {str(e)}",
            "citations": [],
            "confidence": "low"
        }

@app.post("/chat")
async def chat(message: MessageRequest):
    """
    Streaming endpoint for assistant-ui framework.
    Expects: {"content": "user question"}

    Workflow:
    1. User enters message in chat UI
    2. Retriever converts message to vectors and searches Qdrant
    3. Returns top N relevant chunks
    4. ContextAssembler formats chunks into prompt
    5. LLM streams patient-friendly answer in real-time
    6. Streamed answer sent to UI as NDJSON

    Returns streaming response:
    - {"type": "text", "content": "streaming text chunk"}
    - {"type": "end"}
    """
    question = message.content
    logger.info(f"💬 Chat message received: {question}")

    async def generate():
        try:
            # STEP 1: Retrieve relevant chunks from Qdrant
            logger.info("🔍 Retrieving chunks...")
            retrieval_result = retrieve(question)
            text_chunks = retrieval_result.get("text_chunks", [])

            if not text_chunks:
                logger.warning("❌ No chunks found")
                yield json.dumps({"type": "text", "content": "No relevant information found. Please try rewording your question."}) + "\n"
                yield json.dumps({"type": "end"}) + "\n"
                return

            logger.info(f"✅ Retrieved {len(text_chunks)} chunks")

            # STEP 2: Assemble context from chunks
            logger.info("⚙️  Assembling context...")
            assembler = ContextAssembler()
            context = assembler.assemble_context(
                chunks=text_chunks,
                drug_brand="",
                drug_generic="",
                drug_class="",
                query=question
            )

            # STEP 3: Stream answer from LLM
            logger.info("🤖 Streaming answer from LLM...")
            for chunk in assembler.generate_answer_streaming(
                context=context,
                query=question,
                chunks=text_chunks
            ):
                yield json.dumps({"type": "text", "content": chunk}) + "\n"

            logger.info("✅ Answer stream complete")
            # Send end signal
            yield json.dumps({"type": "end"}) + "\n"

        except Exception as e:
            logger.error(f"❌ Error in chat: {str(e)}", exc_info=True)
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
