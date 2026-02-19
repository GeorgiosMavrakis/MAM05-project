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
                yield json.dumps({"type": "text", "content": "No relevant information found in the database. Please try a different question."}) + "\n"
                yield json.dumps({"type": "end"}) + "\n"
                return

            logger.info(f"✅ Retrieved {len(text_chunks)} chunks")

            # STEP 2: Assemble context from chunks
            logger.info("⚙️  Assembling context...")
            try:
                assembler = ContextAssembler()
                context = assembler.assemble_context(
                    chunks=text_chunks,
                    drug_brand="",
                    drug_generic="",
                    drug_class="",
                    query=question
                )
            except Exception as e:
                logger.error(f"❌ Error assembling context: {str(e)}", exc_info=True)
                yield json.dumps({"type": "error", "content": f"Error preparing response: {str(e)}"}) + "\n"
                return

            # STEP 3: Stream answer from LLM
            logger.info("🤖 Streaming answer from LLM...")
            answer_chunk_count = 0
            error_occurred = False
            error_message = None

            try:
                for chunk in assembler.generate_answer_streaming(
                    context=context,
                    query=question,
                    chunks=text_chunks
                ):
                    # Check if chunk contains error indicators
                    is_error = isinstance(chunk, str) and (
                        chunk.startswith("Error:") or
                        chunk.startswith("API Error") or
                        "APIConnectionError" in chunk or
                        "Connection error" in chunk
                    )

                    if is_error:
                        logger.warning(f"❌ LLM error detected: {chunk[:100]}")
                        error_occurred = True
                        error_message = chunk
                        # Prepend error marker if not already present
                        if not chunk.startswith("❌"):
                            chunk = f"❌ {chunk}"
                        logger.info(f"Sending error to frontend: {chunk[:100]}")
                        yield json.dumps({"type": "text", "content": chunk}) + "\n"
                        break  # Stop processing on error
                    else:
                        answer_chunk_count += 1
                        logger.debug(f"[Chunk {answer_chunk_count}] {chunk[:50]}")
                        yield json.dumps({"type": "text", "content": chunk}) + "\n"
            except Exception as e:
                logger.error(f"❌ Error during streaming: {str(e)}", exc_info=True)
                error_occurred = True
                error_message = str(e)
                error_text = f"❌ Error generating response: {str(e)}"
                logger.info(f"Sending exception error to frontend: {error_text[:100]}")
                yield json.dumps({"type": "text", "content": error_text}) + "\n"

            if error_occurred:
                logger.warning(f"❌ Chat ended with error: {error_message}")
            else:
                logger.info(f"✅ Answer stream complete ({answer_chunk_count} chunks)")

            # Send end signal
            yield json.dumps({"type": "end"}) + "\n"

        except Exception as e:
            logger.error(f"❌ Unexpected error in chat: {str(e)}", exc_info=True)
            try:
                yield json.dumps({"type": "error", "content": f"Unexpected error: {str(e)}"}) + "\n"
            except:
                pass

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "Medical RAG API"}

@app.get("/test-llm")
async def test_llm():
    """Test endpoint to verify LLM connectivity."""
    try:
        from context_assembly import ContextAssembler
        assembler = ContextAssembler()

        # Try a simple request
        result = "Starting LLM test...\n"
        result += f"Endpoint: {assembler.endpoint}\n"
        result += f"API Key present: {'Yes' if assembler.api_key else 'No'}\n"

        # Try sending a simple request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {assembler.api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Say 'Hello'"}
            ],
            "temperature": 0.2,
            "max_tokens": 10,
            "stream": False
        }

        import requests
        response = requests.post(
            assembler.endpoint,
            json=payload,
            headers=headers,
            timeout=15
        )

        result += f"Response Status: {response.status_code}\n"
        result += f"Response Text: {response.text[:200]}\n"

        return {"status": "test_completed", "details": result}
    except Exception as e:
        return {"status": "error", "error": str(e), "details": str(type(e))}

if __name__ == "__main__":
    uvicorn.run(
        "API:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
