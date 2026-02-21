"""
STEP 6 & 7: CONTEXT ASSEMBLY AND LLM ANSWER GENERATION
Formats retrieved chunks for LLM and generates structured answers with citations.

STEP 6: Context Assembly
- Organizes chunks by drug and category
- Formats information hierarchically
- Maintains source tracking for citations

STEP 7: LLM Answer Generation
- Sends formatted context to GPT-4
- Generates patient-friendly answers
- Extracts and structures citations
- Provides confidence scoring
"""

from typing import List, Dict, Optional, Any
import json
import requests
from dotenv import load_dotenv
import os
from dataclasses import dataclass
import logging

from retriever import retrieve

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
logger.info(f"[ENV] Looking for .env at: {env_path}")
logger.info(f"[ENV] .env file exists: {os.path.exists(env_path)}")

load_dotenv(dotenv_path=env_path, override=True)
API_KEY = os.getenv("API_KEY")

logger.info(f"[ENV] API_KEY loaded: {bool(API_KEY)}")
if API_KEY:
    logger.info(f"[ENV] API_KEY first 20 chars: {API_KEY[:20]}...")
else:
    logger.warning("[ENV] WARNING: API_KEY is not set!")


@dataclass
class Citation:
    """Represents a citation to a source chunk."""
    chunk_id: int
    drug: str
    category: str
    source_text: str
    relevance_score: float


@dataclass
class LLMAnswer:
    """Structured answer from LLM with citations."""
    answer: str
    citations: List[Citation]
    confidence: str  # "high", "medium", "low"
    answer_category: Optional[str]  # What aspect of drug was answered


class ContextAssembler:
    """Assembles context from retrieved chunks for LLM processing."""

    def __init__(self):
        """Initialize the context assembler."""
        self.api_key = API_KEY
        self.endpoint = "https://llmproxy.uva.nl/v1/chat/completions"

        # Validation and logging
        logger.info(f"[ContextAssembler] Initializing...")
        logger.info(f"[ContextAssembler] API Key set: {bool(self.api_key)}")
        logger.info(f"[ContextAssembler] Endpoint: {self.endpoint}")

        if not self.api_key:
            logger.error("[ContextAssembler] ERROR: API_KEY is not set!")
            raise ValueError("API_KEY environment variable is not set")
        if not self.endpoint:
            logger.error("[ContextAssembler] ERROR: Endpoint is not configured!")
            raise ValueError("Endpoint is not configured")

        logger.info(f"[ContextAssembler] Successfully initialized")

    def assemble_context(
        self,
        chunks: List[Any],
        drug_brand: str,
        drug_generic: str,
        drug_class: str,
        query: str
    ) -> str:
        """
        STEP 6: Assemble retrieved chunks into structured context for LLM.

        Args:
            chunks: List of RetrievedChunk objects from retriever
            drug_brand: Brand name of drug
            drug_generic: Generic name of drug
            drug_class: Drug class/category
            query: Original user query

        Returns:
            Formatted context string suitable for LLM
        """
        if not chunks:
            return "No relevant information found in database."

        # Group chunks by category for better organization
        chunks_by_category = self._group_by_category(chunks)

        # Build formatted context
        context_parts = []

        # Header with drug information
        context_parts.append(self._build_drug_header(
            drug_brand, drug_generic, drug_class
        ))

        # Organize by category with source tracking
        context_parts.append("=== Relevant Information ===\n")

        for category, category_chunks in sorted(chunks_by_category.items()):
            context_parts.append(f"\n[Source: {category.replace('_', ' ').title()}]")

            for i, chunk in enumerate(category_chunks, 1):
                # Handle both dict and object formats
                if isinstance(chunk, dict):
                    chunk_id = chunk.get("id", f"chunk_{i}")
                    chunk_text = chunk.get("text", "")
                    chunk_score = chunk.get("score", 0.5)
                else:
                    chunk_id = chunk.id
                    chunk_text = chunk.text
                    chunk_score = chunk.similarity_score

                # Include chunk ID for citation tracking
                context_parts.append(f"\n({i}) [Chunk ID: {chunk_id}]")
                context_parts.append(f"Score: {chunk_score:.2f}")

                # Add chunk text with truncation if needed
                text = chunk_text[:500]  # Limit to 500 chars per chunk
                if len(chunk_text) > 500:
                    text += "..."
                context_parts.append(text)

        # Add metadata context
        context_parts.append(self._build_metadata_context(chunks))

        return "\n".join(context_parts)

    def generate_answer_with_citations(
        self,
        context: str,
        query: str,
        chunks: List[Any],
        system_prompt: Optional[str] = None
    ) -> LLMAnswer:
        """
        STEP 7: Send context to LLM and generate structured answer with citations.

        Args:
            context: Formatted context from Step 6
            query: Original user question
            chunks: Original chunk objects for citation building
            system_prompt: Optional custom system prompt

        Returns:
            LLMAnswer object with answer, citations, and confidence
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Build the user prompt with extraction instructions
        user_prompt = self._build_user_prompt(context, query)

        try:
            # Call LLM
            response_text = self._call_llm(system_prompt, user_prompt)

            # Parse response to extract answer and citations
            answer, citations, confidence = self._parse_llm_response(
                response_text, chunks, query
            )

            return LLMAnswer(
                answer=answer,
                citations=citations,
                confidence=confidence,
                answer_category=self._infer_category(query)
            )

        except Exception as e:
            # Fallback answer on error
            return LLMAnswer(
                answer=f"Error generating answer: {str(e)}",
                citations=[],
                confidence="low",
                answer_category=None
            )

    def generate_answer_streaming(
        self,
        context: str,
        query: str,
        chunks: List[Any],
        system_prompt: Optional[str] = None
    ):
        """
        Stream LLM response for real-time UI updates.

        Yields:
            Streaming chunks of text
        """
        import logging
        logger = logging.getLogger(__name__)

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        user_prompt = self._build_user_prompt(context, query)

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "stream": True
            }

            logger.info(f"[LLM] Preparing request...")
            logger.info(f"[LLM] Endpoint: {self.endpoint}")
            logger.info(f"[LLM] Model: {payload['model']}")
            logger.info(f"[LLM] System prompt length: {len(system_prompt)}")
            logger.info(f"[LLM] User prompt length: {len(user_prompt)}")
            logger.info(f"[LLM] Total payload size: {len(json.dumps(payload))} bytes")
            logger.info(f"[LLM] Sending streaming request...")

            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                stream=True,
                timeout=60
            )

            logger.info(f"[LLM] Response status: {response.status_code}")
            response.raise_for_status()

            # Stream response tokens
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                    if line_str.startswith('data: '):
                        try:
                            chunk_data = json.loads(line_str[6:])
                            if chunk_data.get('choices'):
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            pass

        except requests.exceptions.HTTPError as e:
            logger.error(f"[LLM] HTTP Error: {e.response.status_code}")
            logger.error(f"[LLM] Response text: {e.response.text}")
            logger.error(f"[LLM] Request URL: {e.response.url}")
            logger.error(f"[LLM] Request headers: {e.response.request.headers}")
            logger.error(f"[LLM] Request body size: {len(e.response.request.body) if e.response.request.body else 0}")
            # Return the actual error response from the API
            try:
                error_data = e.response.json()
                error_msg = error_data.get('error', {}).get('message', str(error_data))
                yield f"API Error {e.response.status_code}: {error_msg}"
            except:
                yield f"API Error {e.response.status_code}: {e.response.reason}"
        except requests.exceptions.Timeout as e:
            logger.error(f"[LLM] Request timeout: {str(e)}")
            yield "Error: Request timeout - LLM took too long to respond"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[LLM] Connection error: {str(e)}")
            yield "Error: Cannot connect to LLM service"
        except Exception as e:
            logger.error(f"[LLM] Unexpected error: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"

    # ═════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═════════════════════════════════════════════════════════════

    def _group_by_category(self, chunks: List[Any]) -> Dict[str, List[Any]]:
        """Group chunks by their metadata category."""
        grouped = {}
        for chunk in chunks:
            # Handle both dict and object formats
            if isinstance(chunk, dict):
                metadata = chunk.get("metadata", {})
                category = metadata.get("category", "general")
            else:
                metadata = getattr(chunk, 'metadata', {})
                category = metadata.get("category", "general")

            if category not in grouped:
                grouped[category] = []
            grouped[category].append(chunk)
        return grouped

    def _build_drug_header(
        self,
        drug_brand: str,
        drug_generic: str,
        drug_class: str
    ) -> str:
        """Build drug information header."""
        parts = []

        if drug_brand:
            parts.append(f"Drug: {drug_brand}")
        if drug_generic:
            parts.append(f"Generic: {drug_generic}")
        if drug_class:
            parts.append(f"Class: {drug_class}")

        header = " | ".join(parts)
        return f"{'='*70}\n{header}\n{'='*70}\n"

    def _build_metadata_context(self, chunks: List[Any]) -> str:
        """Add metadata information for context."""
        parts = ["\n=== Information Quality Metrics ==="]
        parts.append(f"Total chunks retrieved: {len(chunks)}")

        # Extract scores handling both dict and object formats
        scores = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                score = chunk.get("score", 0.5)
            else:
                score = chunk.similarity_score
            scores.append(score)

        if scores:
            avg_score = sum(scores) / len(scores)
            parts.append(f"Average relevance score: {avg_score:.3f}")

            # Count by score tier
            high = sum(1 for s in scores if s >= 0.8)
            medium = sum(1 for s in scores if 0.5 <= s < 0.8)
            low = sum(1 for s in scores if s < 0.5)

            parts.append(f"High confidence chunks: {high}")
            parts.append(f"Medium confidence chunks: {medium}")
            parts.append(f"Lower confidence chunks: {low}")

        return "\n".join(parts)

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for medical Q&A."""
        return """ROLE:
You are a medical information assistant helping patients understand medications using provided medical information only.
OBJECTIVE:
Provide clear, accurate, and patient-friendly explanations that help users understand medication information without giving medical advice.
CORE RULES:
1. When answering: - Be precise and always use the data provided
2. Do not add external medical knowledge or assumptions
3. Avoid giving medical advice - encourage consulting healthcare providers 
4. Encourage consultation with qualified healthcare professionals when decisions or concerns arise
5. Clearly state when information is uncertain, incomplete, or conflicting
6. If sources contradict each other, explicitly call out the contradiction and summarize the differing statements
7. Prioritize patient safety and clarity
COMMUNICATION STYLE:
- Clear, calm, friendly and reassuring
- Use plain language suitable for non-medical users
SAFETY REQUIREMENTS:
- Prominently highlight warnings, risks, and precautions
- Explicitly note when information differs between medications
- If the provided information is insufficient to answer safely, say so clearly
OUTPUT FORMAT:
Structure responses using the following sections when applicable:
"""

    def _build_user_prompt(self, context: str, query: str) -> str:
        """Build the full user prompt for the LLM."""
        return f"""Based on the following medical information, answer the user's question.

{context}

User Question:
{query}

Please provide:
1. **Answer**
   - Clear explanation addressing the question
2. **Important Safety Information**
   - Warnings, risks, precautions (highlight critical items)
3. **Differences Between Medications** (if applicable)
4. **Uncertainty or Limitations**
   - Missing or conflicting information
5. **When to Contact a Healthcare Provider**
   - Neutral guidance encouraging professional consultation


Remember: Base your answer ONLY on the information provided above."""

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API and return response text."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,  # Lower temperature for consistency
            "max_tokens": 1500
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    def _parse_llm_response(
        self,
        response_text: str,
        chunks: List[Any],
        query: str
    ) -> tuple[str, List[Citation], str]:
        """
        Parse LLM response and extract answer with citations.

        Returns:
            Tuple of (answer_text, citations, confidence)
        """
        # Extract confidence from response or infer from scores
        confidence = self._infer_confidence(chunks)

        # Build citations from chunks with high relevance
        citations = self._build_citations(chunks)

        # The answer is the entire response (LLM generated text)
        answer = response_text

        return answer, citations, confidence

    def _infer_confidence(self, chunks: List[Any]) -> str:
        """Infer confidence level from chunk scores."""
        if not chunks:
            return "low"

        scores = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                score = chunk.get("score", 0.5)
            else:
                score = chunk.similarity_score
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score >= 0.7:
            return "high"
        elif avg_score >= 0.5:
            return "medium"
        else:
            return "low"

    def _build_citations(self, chunks: List[Any]) -> List[Citation]:
        """Convert chunks to Citation objects."""
        citations = []

        for chunk in chunks:
            # Extract score
            if isinstance(chunk, dict):
                score = chunk.get("score", 0.5)
                chunk_id = chunk.get("id", "unknown")
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
            else:
                score = chunk.similarity_score
                chunk_id = chunk.id
                text = chunk.text
                metadata = getattr(chunk, 'metadata', {})

            # Only include high-relevance chunks in citations
            if score < 0.4:
                continue

            citation = Citation(
                chunk_id=chunk_id,
                drug=metadata.get("drug_name_brand", "Unknown"),
                category=metadata.get("category", "general"),
                source_text=text[:200],  # First 200 chars
                relevance_score=score
            )
            citations.append(citation)

        # Sort by relevance score
        citations.sort(key=lambda c: c.relevance_score, reverse=True)

        # Return top 5 citations
        return citations[:5]

    def _infer_category(self, query: str) -> Optional[str]:
        """Infer what aspect of medication the query is about."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["side effect", "adverse", "reaction"]):
            return "adverse_reactions"
        elif any(word in query_lower for word in ["dose", "dosage", "take", "how much"]):
            return "dosage"
        elif any(word in query_lower for word in ["pregnancy", "pregnant", "breastfeed"]):
            return "pregnancy"
        elif any(word in query_lower for word in ["warning", "precaution", "caution"]):
            return "warnings"
        elif any(word in query_lower for word in ["interact", "interaction", "combined"]):
            return "interactions"
        else:
            return None


# ═════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═════════════════════════════════════════════════════════════════

def create_assembler() -> ContextAssembler:
    """Create and return a ContextAssembler instance."""
    return ContextAssembler()


if __name__ == "__main__":
    print("Context Assembly and LLM Answer Generation Module")
    print("Use this with retriever.py to implement STEP 6 & 7")

