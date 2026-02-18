"""
LLM CLIENT - RAG ANSWER GENERATION
Orchestrates retrieval and LLM-based answer generation for medical questions.

STEP 4: Query extraction (external - uses LLM to extract drug/category info)
STEP 5: Retrieval using retriever.py with Qdrant
STEP 6: Context assembly from context_assembly.py
STEP 7: LLM answer generation with citations

Purpose:
- Take patient query
- Extract drug/category info
- Retrieve relevant chunks from Qdrant
- Format context hierarchically
- Generate structured answers with citations
"""
from typing import Dict, Any
import requests
from dotenv import load_dotenv
import os
import json

# Import retriever and context assembler
from retriever import create_retriever
from context_assembly import create_assembler, LLMAnswer

load_dotenv()
API_KEY = os.getenv("API_KEY")
ENDPOINT = "https://ai-research-proxy.azurewebsites.net/v1/chat/completions"


# ═════════════════════════════════════════════════════════════
# STEP 4: QUERY EXTRACTION (Extract drug/category from question)
# ═════════════════════════════════════════════════════════════

def extract_query_intent(question: str) -> Dict[str, Any]:
    """
    STEP 4: Use LLM to extract structured information from query.

    Args:
        question: Patient's natural language question

    Returns:
        Dictionary with:
        {
            "drug_brand": str or None,
            "drug_generic": str or None,
            "drug_class": str or None,
            "categories": List[str],
            "confidence": "low" | "medium" | "high",
            "raw_response": str
        }
    """
    system_prompt = """You are an expert at extracting medication information from patient questions.

Extract:
1. Drug brand name (e.g., Ozempic, Metformin)
2. Drug generic name (e.g., semaglutide, metformin hydrochloride)
3. Drug class (e.g., GLP-1 Receptor Agonist, SGLT2 Inhibitor)
4. Question categories: warnings, dosage, side_effects, interactions, precautions, instructions_for_use, pregnancy
5. Confidence level: high (clear drug specified), medium (partial info), low (unclear/no drug)

Respond ONLY with valid JSON, no markdown or extra text:
{
    "drug_brand": "Brand Name or null",
    "drug_generic": "generic name or null",
    "drug_class": "Drug Class or null",
    "categories": ["category1", "category2"],
    "confidence": "high/medium/low",
    "reasoning": "brief explanation"
}"""

    user_prompt = f"""Extract medication information from this question:

"{question}"

Respond with ONLY valid JSON."""

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }

        response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]

        # Parse JSON response
        extracted = json.loads(response_text)

        # Normalize None values
        if extracted.get("drug_brand") in [None, "null", "None"]:
            extracted["drug_brand"] = None
        if extracted.get("drug_generic") in [None, "null", "None"]:
            extracted["drug_generic"] = None
        if extracted.get("drug_class") in [None, "null", "None"]:
            extracted["drug_class"] = None

        # Ensure categories is a list
        if not isinstance(extracted.get("categories"), list):
            extracted["categories"] = []

        return extracted

    except (json.JSONDecodeError, KeyError, requests.exceptions.RequestException) as e:
        # Fallback on error
        return {
            "drug_brand": None,
            "drug_generic": None,
            "drug_class": None,
            "categories": [],
            "confidence": "low",
            "error": str(e)
        }


# ═════════════════════════════════════════════════════════════
# MAIN RAG PIPELINE
# ═════════════════════════════════════════════════════════════

def ask_rag(question: str) -> str:
    """
    Complete RAG pipeline: Extract → Retrieve → Assemble → Generate

    Args:
        question: Patient's question

    Returns:
        Patient-friendly answer with citations
    """
    try:
        # STEP 4: Extract query intent
        print(f"\n📝 Extracting query intent...")
        extracted = extract_query_intent(question)
        print(f"  Drug: {extracted.get('drug_brand')} ({extracted.get('drug_generic')})")
        print(f"  Class: {extracted.get('drug_class')}")
        print(f"  Categories: {extracted.get('categories')}")
        print(f"  Confidence: {extracted.get('confidence')}")

        # STEP 5: Retrieve chunks from Qdrant
        print(f"\n🔍 Retrieving relevant information...")
        retriever = create_retriever()
        retrieval_result = retriever.retrieve(
            extracted_dict=extracted,
            original_query=question
        )

        # Check if retrieval succeeded
        if retrieval_result.result_type != "retrieved_chunks":
            print(f"  Result type: {retrieval_result.result_type}")
            return retrieval_result.message or "Unable to find relevant information."

        if not retrieval_result.chunks:
            return "No relevant information found in the database. Please try rewording your question."

        print(f"  Retrieved {len(retrieval_result.chunks)} chunks")
        print(f"  Top score: {retrieval_result.chunks[0].similarity_score:.3f}")

        # STEP 6 & 7: Assemble context and generate answer
        print(f"\n⚙️  Assembling context and generating answer...")
        assembler = create_assembler()

        context = assembler.assemble_context(
            chunks=retrieval_result.chunks,
            drug_brand=extracted.get("drug_brand") or "Unknown",
            drug_generic=extracted.get("drug_generic") or "Unknown",
            drug_class=extracted.get("drug_class") or "Unknown",
            query=question
        )

        llm_answer = assembler.generate_answer_with_citations(
            context=context,
            query=question,
            chunks=retrieval_result.chunks
        )

        print(f"  Confidence: {llm_answer.confidence}")
        print(f"  Citations: {len(llm_answer.citations)}")

        # Format final answer with citations
        final_answer = _format_answer_with_citations(llm_answer)

        return final_answer

    except Exception as e:
        return f"Error processing your question: {str(e)}"


def ask_rag_streaming(question: str):
    """
    Stream RAG pipeline response for real-time UI updates.

    Args:
        question: Patient's question

    Yields:
        Streaming text chunks
    """
    try:
        # STEP 4: Extract query intent
        extracted = extract_query_intent(question)

        # STEP 5: Retrieve chunks
        retriever = create_retriever()
        retrieval_result = retriever.retrieve(
            extracted_dict=extracted,
            original_query=question
        )

        # Check if retrieval succeeded
        if retrieval_result.result_type != "retrieved_chunks":
            yield retrieval_result.message or "Unable to find relevant information."
            return

        if not retrieval_result.chunks:
            yield "No relevant information found."
            return

        # STEP 6 & 7: Stream answer
        assembler = create_assembler()

        context = assembler.assemble_context(
            chunks=retrieval_result.chunks,
            drug_brand=extracted.get("drug_brand") or "Unknown",
            drug_generic=extracted.get("drug_generic") or "Unknown",
            drug_class=extracted.get("drug_class") or "Unknown",
            query=question
        )

        # Stream from LLM
        for chunk in assembler.generate_answer_streaming(
            context=context,
            query=question,
            chunks=retrieval_result.chunks
        ):
            yield chunk

    except Exception as e:
        yield f"Error: {str(e)}"


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def _format_answer_with_citations(llm_answer: LLMAnswer) -> str:
    """Format the LLM answer with citations for display."""
    parts = []

    # Main answer
    parts.append(llm_answer.answer)

    # Add citations section if available
    if llm_answer.citations:
        parts.append("\n" + "="*60)
        parts.append("SOURCES:")
        parts.append("="*60)

        for i, citation in enumerate(llm_answer.citations, 1):
            parts.append(f"\n[{i}] {citation.drug} - {citation.category.replace('_', ' ').title()}")
            parts.append(f"    Relevance: {citation.relevance_score:.1%}")
            parts.append(f"    '{citation.source_text}...'")

    # Add confidence note
    parts.append("\n" + "="*60)
    parts.append(f"Information Confidence: {llm_answer.confidence.upper()}")
    parts.append("="*60)

    parts.append("\n⚠️  IMPORTANT: This information is for educational purposes only.")
    parts.append("Always consult with your healthcare provider for medical advice.")

    return "\n".join(parts)
