from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from retriever import retrieve
from context_assembly import ContextAssembler
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class MessageRequest(BaseModel):
    content: str
    messages: List[Dict[str, str]] = []  # Chat history: [{"role": "user", "content": "..."}, ...]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _build_context_aware_query(current_message: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Build a context-aware query by incorporating relevant chat history.
    This helps the retriever understand context even when the current message is vague.

    Example:
    - User: "Tell me about metformin"
    - Assistant: "Metformin is a medication..."
    - User: "any popular brand names?" (vague without context)
    - Result: "Metformin brand names" (contextual enhancement)
    """
    if not chat_history or len(chat_history) == 0:
        logger.info(f"No chat history available, using original query: '{current_message}'")
        return current_message

    logger.info(f"💬 Chat history has {len(chat_history)} messages")

    # Extract PREVIOUS user messages from chat history (exclude the last message which is the current one)
    # The current message should already be in the history as the last user message
    user_messages = []
    for i, msg in enumerate(chat_history):
        if msg.get("role") == "user" and msg.get("content", "").strip():
            content = msg.get("content", "").strip()
            # Skip if this is the current message (don't extract context from what we're about to enhance)
            is_current_msg = (content == current_message.strip())
            if not is_current_msg:
                user_messages.append(content)

    logger.info(f"Found {len(user_messages)} PREVIOUS user messages in history (excluding current)")

    enhanced_query = current_message

    # If current message is short/vague and we have previous context, enhance it
    if len(user_messages) > 0 and len(current_message) < 100:
        # Combine all previous user messages to extract context
        all_context = " ".join(user_messages)
        logger.info(f"📖 Full chat context: {all_context[:150]}...")

        # Extract key terms - look for capitalized words (likely drug names) and longer words
        words = all_context.split()
        context_terms = []

        # Extended stop words list
        stop_words = {
            "tell", "about", "what", "which", "more", "please", "thank",
            "can", "give", "examples", "information", "side", "effects",
            "dosage", "taking", "with", "for", "and", "the", "are", "is",
            "you", "me", "how", "why", "when", "where", "do", "does", "did",
            "have", "has", "been", "being", "be", "to", "of", "in", "on", "at",
            "by", "from", "as", "or", "an", "a", "any", "all", "each", "every",
            "some", "many", "few", "other", "another", "this", "that", "these",
            "those", "i", "it", "he", "she", "we", "they", "them", "my", "your",
            "explain", "describe", "talk", "discuss", "mention", "list", "show",
            "provide", "help", "assist", "need", "want", "would", "could", "should"
        }

        for word in words:
            # Skip very short words and common stop words
            if len(word) > 3:
                clean_word = word.strip('.,!?;:').lower()
                # Skip if it's in stop words list
                if clean_word and clean_word not in stop_words:
                    context_terms.append(word.strip('.,!?;:'))  # Keep original case

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in context_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)

        if unique_terms:
            # Take the top 3 most relevant terms (not 5, to avoid noise)
            context_prefix = " ".join(unique_terms[:3])
            enhanced_query = f"{context_prefix} {current_message}".strip()
            logger.info(f"✅ Enhanced query: '{enhanced_query}'")
        else:
            logger.info(f"⚠️  No context terms extracted, using original: '{current_message}'")
    elif len(current_message) >= 100:
        logger.info(f"Message is detailed enough, using original query")

    return enhanced_query

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
    Streaming endpoint for assistant-ui framework with chat history support.
    Expects: {"content": "user question", "messages": [...chat history...]}

    Workflow:
    1. Builds context-aware query from chat history and current message
    2. Retriever converts query to vectors and searches Qdrant
    3. Returns top N relevant chunks
    4. ContextAssembler formats chunks into prompt with conversation context
    5. LLM streams patient-friendly answer in real-time
    6. Streamed answer sent to UI as NDJSON

    Returns streaming response:
    - {"type": "text", "content": "streaming text chunk"}
    - {"type": "end"}
    """
    question = message.content
    chat_history = message.messages or []

    logger.info(f"💬 Chat message received: {question}")
    logger.info(f"📚 Chat history length: {len(chat_history)}")

    # Log and filter each message in the history - remove empty ones
    filtered_history = []
    for idx, msg in enumerate(chat_history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        if content:  # Only keep non-empty messages
            filtered_history.append({"role": role, "content": content})
            logger.info(f"  [{idx}] {role}: {content[:100]}{'...' if len(content) > 100 else ''}")
        else:
            logger.info(f"  [{idx}] {role}: [EMPTY - FILTERED OUT]")

    chat_history = filtered_history
    logger.info(f"📚 Filtered history length: {len(chat_history)} (removed {len(message.messages) - len(chat_history)} empty messages)")

    async def generate():
        end_sent = False
        try:
            # Build context-aware query from chat history
            context_aware_query = _build_context_aware_query(question, chat_history)
            logger.info(f"🔄 Context-aware query: {context_aware_query[:100]}...")

            # STEP 1: Retrieve relevant chunks from Qdrant
            logger.info("🔍 Retrieving chunks...")
            retrieval_result = retrieve(context_aware_query)
            text_chunks = retrieval_result.get("text_chunks", [])

            if not text_chunks:
                logger.warning("❌ No chunks found")
                yield json.dumps({"type": "text", "content": "No relevant information found in the database. Please try a different question."}) + "\n"
                yield json.dumps({"type": "end"}) + "\n"
                end_sent = True
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
                    query=context_aware_query  # Use enhanced query for better context formatting
                )
            except Exception as e:
                logger.error(f"❌ Error assembling context: {str(e)}", exc_info=True)
                yield json.dumps({"type": "error", "content": f"Error preparing response: {str(e)}"}) + "\n"
                yield json.dumps({"type": "end"}) + "\n"
                end_sent = True
                return

            # STEP 3: Stream answer from LLM with conversation history
            logger.info("🤖 Streaming answer from LLM...")
            answer_chunk_count = 0
            error_occurred = False
            error_message = None

            try:
                for chunk in assembler.generate_answer_streaming_with_history(
                    context=context,
                    query=context_aware_query,  # Use enhanced query so LLM knows the context
                    chunks=text_chunks,
                    chat_history=chat_history
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

                # Log source references for troubleshooting (not sent to UI)
                logger.info(f"📚 Sources used ({len(text_chunks)} total):")
                for i, chunk in enumerate(text_chunks[:10], 1):
                    chunk_id = chunk.get("id", "unknown")
                    chunk_text = chunk.get("text", "")[:150]
                    logger.info(f"  {i}. {chunk_id}: {chunk_text}...")

                # Sources are now only logged, not sent to frontend


        except Exception as e:
            logger.error(f"❌ Unexpected error in chat: {str(e)}", exc_info=True)
            try:
                yield json.dumps({"type": "error", "content": f"Unexpected error: {str(e)}"}) + "\n"
            except:
                pass
        finally:
            # ALWAYS send end signal to stop loading indicator (if not already sent)
            if not end_sent:
                logger.info("📤 Sending final end signal")
                yield json.dumps({"type": "end"}) + "\n"

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
