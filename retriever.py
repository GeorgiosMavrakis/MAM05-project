"""
RETRIEVER MODULE - BETA
Handles retrieval pipeline for FDA drug information RAG system.

Features:
- Scenario-based routing (A, B, C, D)
- Metadata filtering with auto-append categories
- Configurable retrieval limits
- Optional reranking (toggle on/off)
- Progressive fallback for empty results
- Multi-drug support
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest
)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Retrieval limits
INITIAL_RETRIEVAL_LIMIT = 70   # Chunks retrieved from Qdrant
FINAL_RERANK_LIMIT = 30         # Chunks after reranking (or top-K by score if reranker off)

# Reranker toggle
RERANKER_ON = True  # Set to False to disable reranking (uses similarity scores only)

# Score thresholds
MIN_SIMILARITY_SCORE = 0.3  # Minimum cosine similarity to keep chunk
MIN_RERANK_SCORE = 0.2      # Minimum rerank score to keep chunk (if reranker on)

# Auto-append categories (when drug is specified)
AUTO_APPEND_CATEGORIES = [
    "instructions_for_use",
    "spl_patient_package_insert",
    "spl_unclassified_section",
    "precautions",
    "warnings_and_cautions",
    "spl_medguide"
]

# Clarification settings
MAX_SUGGESTIONS_LOW_CONFIDENCE = 4
FUZZY_MATCH_THRESHOLD_LOW = 0.3
FUZZY_MATCH_THRESHOLD_MEDIUM = 0.7

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fda_drugs"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with scores and metadata."""
    id: int
    text: str
    similarity_score: float
    rerank_score: Optional[float]
    metadata: Dict[str, Any]
    original_id: str


@dataclass
class RetrievalResult:
    """Result returned by retriever."""
    result_type: str  # "retrieved_chunks", "off_topic", "clarification_needed", etc.
    chunks: Optional[List[RetrievedChunk]]
    message: Optional[str]
    metadata: Optional[Dict[str, Any]]
    fallback_type: Optional[str]
    awaiting_confirmation: bool = False
    suggested_dict: Optional[Dict] = None


# ═══════════════════════════════════════════════════════════════
# RESPONSE TEMPLATES
# ═══════════════════════════════════════════════════════════════

TEMPLATE_A = """
I specialize in answering questions about diabetes medications only.

I can help you with:
- Specific medications (e.g., Ozempic, Metformin, Jardiance)
- Drug classes (e.g., GLP-1 agonists, SGLT2 inhibitors)
- Topics like dosage, side effects, warnings, and interactions

Is there a diabetes medication you'd like to learn about?
"""

TEMPLATE_A1_SUFFIX = """

Note: For medication-specific questions about diabetes drugs, please ask about 
particular medications like Ozempic, Metformin, or Jardiance.
"""

TEMPLATE_B = """
{drug_class} are used for diabetes management.

{predefined_description}

Medications in this class include:
{drug_list}

Would you like specific information about any of these medications?
You can ask about dosage, side effects, warnings, or other topics.
"""

TEMPLATE_B_SPECIFIC = """
Regarding {categories} for {drug_class}:

{predefined_info}

Note: Specific information may vary between medications in this class.
Would you like information about a specific drug?

Available medications: {drug_list}
"""

TEMPLATE_C_LOW = """
I couldn't identify this medication with certainty.
Did you mean one of these:

{suggestions}

Please select or specify the medication you're asking about.
"""

TEMPLATE_C_MEDIUM = """
Did you mean {suggested_drug_brand} ({suggested_drug_generic})?

Please confirm:
- Yes, that's correct
- No, I meant a different medication [please specify]
"""


# ═══════════════════════════════════════════════════════════════
# RETRIEVER CLASS
# ═══════════════════════════════════════════════════════════════

class Retriever:
    """
    Main retriever class for FDA drug information.
    Handles all retrieval scenarios and filtering logic.
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
        predefined_answers: Optional[Dict] = None
    ):
        """
        Initialize retriever with models and connections.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name in Qdrant
            embedding_model: Sentence transformer model name
            reranker_model: Cross-encoder model name
            predefined_answers: Dictionary of predefined class answers
        """
        print("Initializing Retriever...")

        # Connect to Qdrant
        print(f"  Connecting to Qdrant at {qdrant_url}...")
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

        # Load embedding model
        print(f"  Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Load reranker (if enabled)
        if RERANKER_ON:
            print(f"  Loading reranker model: {reranker_model}...")
            self.reranker = CrossEncoder(reranker_model)
        else:
            print("  Reranker DISABLED (will use similarity scores only)")
            self.reranker = None

        # Store predefined answers
        self.predefined_answers = predefined_answers or {}

        print("✓ Retriever initialized successfully")
        print(f"  Reranker: {'ON' if RERANKER_ON else 'OFF'}")
        print(f"  Initial retrieval limit: {INITIAL_RETRIEVAL_LIMIT}")
        print(f"  Final limit: {FINAL_RERANK_LIMIT}")

    # ═══════════════════════════════════════════════════════════
    # MAIN RETRIEVAL FUNCTION
    # ═══════════════════════════════════════════════════════════
    def retrieve(
        self,
        extracted_dict: Dict,
        original_query: str
    ) -> RetrievalResult:
        """
        Main retrieval function - routes to appropriate scenario.

        Args:
            extracted_dict: Dictionary from LLM extraction with:
                - drug_brand: str or None
                - drug_generic: str or None
                - drug_class: str or None
                - categories: List[str]
                - confidence: "low" | "medium" | "high"
            original_query: Original user query text

        Returns:
            RetrievalResult with chunks and metadata
        """

        # Extract fields
        drug_brand = extracted_dict.get("drug_brand")
        drug_generic = extracted_dict.get("drug_generic")
        drug_class = extracted_dict.get("drug_class")
        categories = extracted_dict.get("categories", [])
        confidence = extracted_dict.get("confidence", "high")

        # Handle list inputs for multi-drug queries
        if isinstance(drug_brand, list):
            return self._handle_multi_drug(extracted_dict, original_query)

        # ───────────────────────────────────────────────────────
        # SCENARIO A: Nothing extracted
        # ───────────────────────────────────────────────────────

        if not drug_brand and not drug_generic and not drug_class:

            # Check if medical but off-topic (has categories but no drug)
            if categories and len(categories) > 0:
                # CASE A1: Try pure semantic search
                return self._handle_semantic_search(original_query)

            # CASE A: Completely off-topic
            return RetrievalResult(
                result_type="off_topic",
                chunks=None,
                message=TEMPLATE_A,
                metadata=None,
                fallback_type=None
            )

        # ───────────────────────────────────────────────────────
        # SCENARIO B: Drug class only
        # ───────────────────────────────────────────────────────

        if not drug_brand and not drug_generic and drug_class:

            # Check for multiple classes (B2)
            if isinstance(drug_class, list):
                return self._handle_class_comparison(drug_class)

            # Single class (B1)
            return self._handle_class_query(drug_class, categories)

        # ───────────────────────────────────────────────────────
        # SCENARIO C: Low/Medium confidence
        # ───────────────────────────────────────────────────────

        if confidence == "low":
            return self._handle_low_confidence(extracted_dict)

        if confidence == "medium":
            return self._handle_medium_confidence(extracted_dict)

        # ───────────────────────────────────────────────────────
        # SCENARIO D: Main retrieval pipeline
        # ───────────────────────────────────────────────────────

        if confidence == "high" and (drug_brand or drug_generic):
            return self._handle_main_retrieval(extracted_dict, original_query)

        # Fallback (shouldn't reach here)
        return RetrievalResult(
            result_type="error",
            chunks=None,
            message="Unexpected input format. Please try again.",
            metadata=None,
            fallback_type=None
        )

    # ═══════════════════════════════════════════════════════════
    # SCENARIO HANDLERS
    # ═══════════════════════════════════════════════════════════
    def _handle_semantic_search(self, query: str) -> RetrievalResult:
        """
        CASE A1: Medical query but no drug identified.
        Try pure semantic search without filters.
        """
        print("\n[CASE A1] Pure semantic search (no metadata filter)")

        # Embed query
        query_vector = self._embed_query(query)

        # Search without any filter
        candidates = self._search_qdrant(
            query_vector=query_vector,
            metadata_filter=None,  # No filter!
            limit=INITIAL_RETRIEVAL_LIMIT
        )

        if not candidates:
            return RetrievalResult(
                result_type="off_topic",
                chunks=None,
                message=TEMPLATE_A,
                metadata=None,
                fallback_type="no_results"
            )

        # Rerank or select top by score
        final_chunks = self._rerank_or_select(query, candidates)

        return RetrievalResult(
            result_type="semantic_search_fallback",
            chunks=final_chunks,
            message=TEMPLATE_A1_SUFFIX,
            metadata={
                "total_candidates": len(candidates),
                "returned": len(final_chunks)
            },
            fallback_type="semantic_only"
        )

    def _handle_class_query(
        self,
        drug_class: str,
        categories: List[str]
    ) -> RetrievalResult:
        """
        CASE B1: Drug class with categories.
        Search by class or return predefined answer.
        """
        print(f"\n[CASE B1] Drug class query: {drug_class}")
        print(f"  Categories: {categories}")

        # Get predefined info if available
        predefined_info = self.predefined_answers.get(drug_class, {})
        drug_list = predefined_info.get("drugs", [])

        # If categories are general/empty, return predefined overview
        if not categories or categories == ["general"]:
            message = TEMPLATE_B.format(
                drug_class=drug_class,
                predefined_description=predefined_info.get("description", ""),
                drug_list=", ".join(drug_list) if drug_list else "N/A"
            )

            return RetrievalResult(
                result_type="class_general",
                chunks=None,
                message=message,
                metadata={"drug_class": drug_class},
                fallback_type=None
            )

        # Otherwise, try to search DB by class + categories
        # (This requires drug_class in metadata, which we have!)
        # For now, return predefined + prompt for specific drug
        message = TEMPLATE_B_SPECIFIC.format(
            categories=", ".join(categories),
            drug_class=drug_class,
            predefined_info=predefined_info.get("category_info", ""),
            drug_list=", ".join(drug_list) if drug_list else "N/A"
        )

        return RetrievalResult(
            result_type="class_specific",
            chunks=None,
            message=message,
            metadata={"drug_class": drug_class, "categories": categories},
            fallback_type=None
        )


    def _handle_class_comparison(self, drug_classes: List[str]) -> RetrievalResult:
        """
        CASE B2: Multiple drug classes.
        Return predefined info for each class for LLM to compare.
        """
        print(f"\n[CASE B2] Multiple drug classes: {drug_classes}")

        predefined_texts = []
        for drug_class in drug_classes:
            info = self.predefined_answers.get(drug_class, {})
            predefined_texts.append({
                "drug_class": drug_class,
                "description": info.get("description", ""),
                "drugs": info.get("drugs", [])
            })

        return RetrievalResult(
            result_type="class_comparison",
            chunks=None,
            message=None,
            metadata={
                "drug_classes": drug_classes,
                "predefined_texts": predefined_texts,
                "instruction_to_llm": "Compare these drug classes"
            },
            fallback_type=None
        )


    def _handle_low_confidence(self, extracted_dict: Dict) -> RetrievalResult:
        """
        CASE C-LOW: Very uncertain about drug.
        Show multiple suggestions.
        """
        print("\n[CASE C-LOW] Low confidence - showing suggestions")

        # In a real system, would use fuzzy matching here
        # For now, use extracted info to create suggestion
        suggestions = self._generate_suggestions(
            extracted_dict,
            max_suggestions=MAX_SUGGESTIONS_LOW_CONFIDENCE
        )

        suggestions_text = "\n".join([
            f"• {s['brand']} ({s['generic']}) - {s['class']}"
            for s in suggestions
        ])

        message = TEMPLATE_C_LOW.format(suggestions=suggestions_text)

        return RetrievalResult(
            result_type="clarification_needed",
            chunks=None,
            message=message,
            metadata=None,
            fallback_type=None,
            awaiting_confirmation=True,
            suggested_dict=extracted_dict
        )


    def _handle_medium_confidence(self, extracted_dict: Dict) -> RetrievalResult:
        """
        CASE C-MEDIUM: Somewhat uncertain.
        Show single best match for confirmation.
        """
        print("\n[CASE C-MEDIUM] Medium confidence - requesting confirmation")

        suggested_brand = extracted_dict.get("drug_brand", "Unknown")
        suggested_generic = extracted_dict.get("drug_generic", "Unknown")

        message = TEMPLATE_C_MEDIUM.format(
            suggested_drug_brand=suggested_brand,
            suggested_drug_generic=suggested_generic
        )

        return RetrievalResult(
            result_type="clarification_needed",
            chunks=None,
            message=message,
            metadata=None,
            fallback_type=None,
            awaiting_confirmation=True,
            suggested_dict=extracted_dict
        )


    def _handle_multi_drug(
        self,
        extracted_dict: Dict,
        original_query: str
    ) -> RetrievalResult:
        """
        CASE D2: Multiple drugs in query.
        Retrieve chunks for each drug separately.
        """
        drug_brands = extracted_dict["drug_brand"]
        print(f"\n[CASE D2] Multi-drug query: {drug_brands}")

        all_chunks = {}
        total_candidates = 0

        for drug_brand in drug_brands:
            print(f"\n  Processing drug: {drug_brand}")

            # Create single-drug dict
            single_drug_dict = extracted_dict.copy()
            single_drug_dict["drug_brand"] = drug_brand

            # Retrieve for this drug
            result = self._handle_main_retrieval(single_drug_dict, original_query)

            if result.chunks:
                all_chunks[drug_brand] = result.chunks
                total_candidates += result.metadata.get("total_candidates", 0)

        return RetrievalResult(
            result_type="multi_drug_retrieval",
            chunks=None,  # Chunks organized by drug
            message=None,
            metadata={
                "drugs": all_chunks,
                "total_candidates": total_candidates,
                "instruction_to_llm": "Compare these drugs based on provided chunks"
            },
            fallback_type=None
        )


    def _handle_main_retrieval(
        self,
        extracted_dict: Dict,
        original_query: str
    ) -> RetrievalResult:
        """
        CASE D: Main retrieval pipeline.
        Full pipeline with filtering, search, and reranking.
        """
        print("\n[CASE D] Main retrieval pipeline")

        drug_brand = extracted_dict.get("drug_brand")
        drug_generic = extracted_dict.get("drug_generic")
        categories = extracted_dict.get("categories", [])

        print(f"  Drug brand: {drug_brand}")
        print(f"  Drug generic: {drug_generic}")
        print(f"  Categories: {categories}")

        # Auto-append high-value categories
        categories_expanded = self._auto_append_categories(
            categories,
            has_drug=(drug_brand is not None or drug_generic is not None)
        )

        print(f"  Categories after auto-append: {len(categories_expanded)}")

        # Update dict with expanded categories
        extracted_dict_expanded = extracted_dict.copy()
        extracted_dict_expanded["categories"] = categories_expanded

        # Embed query
        query_vector = self._embed_query(original_query)

        # Build metadata filter
        metadata_filter = self._build_metadata_filter(extracted_dict_expanded)

        # Search Qdrant
        candidates = self._search_qdrant(
            query_vector=query_vector,
            metadata_filter=metadata_filter,
            limit=INITIAL_RETRIEVAL_LIMIT
        )

        print(f"  Retrieved candidates: {len(candidates)}")

        # Handle empty results with fallback
        if not candidates:
            print("  No results, trying fallback...")
            return self._handle_empty_results(
                query_vector=query_vector,
                extracted_dict=extracted_dict_expanded,
                original_query=original_query
            )

        # Rerank or select top by score
        final_chunks = self._rerank_or_select(original_query, candidates)

        print(f"  Final chunks: {len(final_chunks)}")

        return RetrievalResult(
            result_type="retrieved_chunks",
            chunks=final_chunks,
            message=None,
            metadata={
                "drug_brand": drug_brand,
                "drug_generic": drug_generic,
                "drug_class": extracted_dict.get("drug_class"),
                "categories_searched": categories_expanded,
                "total_candidates": len(candidates),
                "returned": len(final_chunks),
                "reranker_used": RERANKER_ON
            },
            fallback_type=None
        )


    # ═══════════════════════════════════════════════════════════
    # EMPTY RESULTS HANDLING (PROGRESSIVE FALLBACK)
    # ═══════════════════════════════════════════════════════════

    def _handle_empty_results(
        self,
        query_vector: List[float],
        extracted_dict: Dict,
        original_query: str
    ) -> RetrievalResult:
        """
        Progressive fallback strategy when no results found.
        Attempts 4 strategies in order.
        """
        print("\n[FALLBACK] Handling empty results...")

        # ───────────────────────────────────────────────────────
        # ATTEMPT 1: Remove category filter, keep drug filter
        # ───────────────────────────────────────────────────────

        print("  Attempt 1: Removing category filter...")

        filter_no_categories = self._build_metadata_filter(
            extracted_dict,
            skip_categories=True
        )

        candidates = self._search_qdrant(
            query_vector=query_vector,
            metadata_filter=filter_no_categories,
            limit=INITIAL_RETRIEVAL_LIMIT
        )

        if candidates:
            print(f"  ✓ Found {len(candidates)} results without category filter")
            final_chunks = self._rerank_or_select(original_query, candidates)

            return RetrievalResult(
                result_type="retrieved_chunks",
                chunks=final_chunks,
                message=f"I couldn't find specific information about the requested "
                        f"topics, but here's general information about the medication.",
                metadata={
                    "total_candidates": len(candidates),
                    "returned": len(final_chunks)
                },
                fallback_type="no_category_filter"
            )

        # ───────────────────────────────────────────────────────
        # ATTEMPT 2: Use drug_class instead of specific drug
        # ───────────────────────────────────────────────────────

        drug_class = extracted_dict.get("drug_class")

        if drug_class:
            print(f"  Attempt 2: Searching by drug class ({drug_class})...")

            filter_class_only = Filter(
                must=[
                    FieldCondition(
                        key="drug_class",
                        match=MatchValue(value=drug_class)
                    )
                ]
            )

            candidates = self._search_qdrant(
                query_vector=query_vector,
                metadata_filter=filter_class_only,
                limit=INITIAL_RETRIEVAL_LIMIT
            )

            if candidates:
                print(f"  ✓ Found {len(candidates)} results for drug class")
                final_chunks = self._rerank_or_select(original_query, candidates)

                return RetrievalResult(
                    result_type="retrieved_chunks",
                    chunks=final_chunks,
                    message=f"I couldn't find information about the specific medication, "
                            f"but here's information about the drug class ({drug_class}).",
                    metadata={
                        "total_candidates": len(candidates),
                        "returned": len(final_chunks)
                    },
                    fallback_type="class_level"
                )

        # ───────────────────────────────────────────────────────
        # ATTEMPT 3: Pure semantic search (no filter)
        # ───────────────────────────────────────────────────────

        print("  Attempt 3: Pure semantic search (no filter)...")

        candidates = self._search_qdrant(
            query_vector=query_vector,
            metadata_filter=None,
            limit=INITIAL_RETRIEVAL_LIMIT
        )

        if candidates:
            print(f"  ✓ Found {len(candidates)} results via semantic search")
            final_chunks = self._rerank_or_select(original_query, candidates)

            return RetrievalResult(
                result_type="retrieved_chunks",
                chunks=final_chunks,
                message="I found some potentially relevant information, but it may "
                        "not be specific to your query.",
                metadata={
                    "total_candidates": len(candidates),
                    "returned": len(final_chunks)
                },
                fallback_type="semantic_only"
            )

        # ───────────────────────────────────────────────────────
        # ATTEMPT 4: Total failure
        # ───────────────────────────────────────────────────────

        print("  ✗ All fallback attempts failed")

        return RetrievalResult(
            result_type="no_results",
            chunks=None,
            message="""I couldn't find information about this query.

Please try:
- Checking the medication name spelling
- Being more specific about what you'd like to know
- Asking about a different medication""",
            metadata=None,
            fallback_type="no_results"
        )


    # ═══════════════════════════════════════════════════════════
    # CORE RETRIEVAL FUNCTIONS
    # ═══════════════════════════════════════════════════════════

    def _embed_query(self, query: str) -> List[float]:
        """Embed query text to vector."""
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()


    def _build_metadata_filter(
        self,
        extracted_dict: Dict,
        skip_categories: bool = False
    ) -> Optional[Filter]:
        """
        Build Qdrant metadata filter from extracted dictionary.

        Args:
            extracted_dict: Extracted drug and category info
            skip_categories: If True, don't filter by categories

        Returns:
            Qdrant Filter object or None
        """
        must_conditions = []

        # ─────────────────────────────────────────────────────
        # Drug filter (brand OR generic OR class)
        # ─────────────────────────────────────────────────────

        drug_brand = extracted_dict.get("drug_brand")
        drug_generic = extracted_dict.get("drug_generic")
        drug_class = extracted_dict.get("drug_class")

        drug_conditions = []

        if drug_brand:
            drug_conditions.append(
                FieldCondition(
                    key="drug_name_brand",
                    match=MatchValue(value=drug_brand)
                )
            )

        if drug_generic:
            drug_conditions.append(
                FieldCondition(
                    key="drug_name_generic",
                    match=MatchValue(value=drug_generic)
                )
            )

        # If we have specific drug, use it
        if drug_conditions:
            must_conditions.append(
                Filter(should=drug_conditions)
            )
        # Otherwise, if we have drug class, use that
        elif drug_class:
            must_conditions.append(
                FieldCondition(
                    key="drug_class",
                    match=MatchValue(value=drug_class)
                )
            )

        # ─────────────────────────────────────────────────────
        # Category filter (unless skipped)
        # ─────────────────────────────────────────────────────

        if not skip_categories:
            categories = extracted_dict.get("categories", [])

            if categories:
                category_conditions = []

                for category in categories:
                    category_conditions.append(
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category)
                        )
                    )

                must_conditions.append(
                    Filter(should=category_conditions)
                )

        # ─────────────────────────────────────────────────────
        # Combine and return
        # ─────────────────────────────────────────────────────

        if must_conditions:
            return Filter(must=must_conditions)
        else:
            return None

    def _search_qdrant(
        self,
        query_vector: List[float],
        metadata_filter: Optional[Filter],
        limit: int
    ) -> List[Any]:
        """
        Search Qdrant with vector and optional filter.

        Returns:
            List of search results with scores and payloads
        """
        try:
            results = self.qdrant_client.search(  # we have to use query_points instead
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=metadata_filter,
                limit=limit,
                score_threshold=MIN_SIMILARITY_SCORE,
                with_payload=True,
                with_vectors=False
            )
            return results
        except Exception as e:
            print(f"  ✗ Qdrant search error: {str(e)}")
            return []


    def _rerank_or_select(
        self,
        query: str,
        candidates: List[Any]
    ) -> List[RetrievedChunk]:
        """
        Rerank candidates (if RERANKER_ON) or select top by similarity score.

        Args:
            query: Original user query
            candidates: List of search results from Qdrant

        Returns:
            List of RetrievedChunk objects (top N)
        """

        if RERANKER_ON and self.reranker:
            # ─────────────────────────────────────────────────
            # RERANKING ENABLED
            # ─────────────────────────────────────────────────
            print(f"  Reranking {len(candidates)} candidates...")

            # Create query-chunk pairs
            pairs = [
                (query, candidate.payload["text"])
                for candidate in candidates
            ]

            # Predict rerank scores
            raw_scores = self.reranker.predict(pairs)

            # Normalize with sigmoid
            rerank_scores = 1 / (1 + np.exp(-raw_scores))

            # Attach scores to candidates
            for i, candidate in enumerate(candidates):
                candidate.rerank_score = float(rerank_scores[i])

            # Sort by rerank score
            candidates_sorted = sorted(
                candidates,
                key=lambda x: x.rerank_score,
                reverse=True
            )

            # Filter by minimum rerank score and take top N
            candidates_filtered = [
                c for c in candidates_sorted
                if c.rerank_score >= MIN_RERANK_SCORE
            ]

            top_candidates = candidates_filtered[:FINAL_RERANK_LIMIT]

            print(f"  → Top {len(top_candidates)} after reranking")

        else:
            # ─────────────────────────────────────────────────
            # RERANKING DISABLED - Use similarity scores
            # ─────────────────────────────────────────────────
            print(f"  Selecting top {FINAL_RERANK_LIMIT} by similarity score...")

            # Already sorted by similarity from Qdrant
            top_candidates = candidates[:FINAL_RERANK_LIMIT]

            # Set rerank_score to None
            for candidate in top_candidates:
                candidate.rerank_score = None

        # Convert to RetrievedChunk objects
        retrieved_chunks = []

        for candidate in top_candidates:
            chunk = RetrievedChunk(
                id=candidate.id,
                text=candidate.payload["text"],
                similarity_score=candidate.score,
                rerank_score=getattr(candidate, "rerank_score", None),
                metadata={
                    k: v for k, v in candidate.payload.items()
                    if k != "text"
                },
                original_id=candidate.payload.get("original_id", "")
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks


    def _auto_append_categories(
        self,
        categories: List[str],
        has_drug: bool
    ) -> List[str]:
        """
        Auto-append high-value categories if drug is specified.

        Args:
            categories: Original categories from extraction
            has_drug: Whether drug_brand or drug_generic is specified

        Returns:
            Expanded category list
        """
        if not has_drug:
            return categories

        # Make a copy to avoid modifying original
        categories_expanded = categories.copy()

        # Add auto-append categories if not already present
        for category in AUTO_APPEND_CATEGORIES:
            if category not in categories_expanded:
                categories_expanded.append(category)

        return categories_expanded


    # ═══════════════════════════════════════════════════════════
    # HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════

    def _generate_suggestions(
        self,
        extracted_dict: Dict,
        max_suggestions: int = 4
    ) -> List[Dict]:
        """
        Generate drug suggestions for low confidence scenarios.
        In production, would use fuzzy matching on drug names.
        """
        # Placeholder implementation
        # In real system, would query drug database with fuzzy matching

        suggested = extracted_dict.get("drug_brand", "Unknown")

        # Mock suggestions (replace with real fuzzy matching)
        suggestions = [
            {
                "brand": suggested if suggested != "Unknown" else "Ozempic",
                "generic": "semaglutide",
                "class": "GLP-1 Receptor Agonists"
            },
            {
                "brand": "Mounjaro",
                "generic": "tirzepatide",
                "class": "GLP-1/GIP Agonists"
            },
            {
                "brand": "Jardiance",
                "generic": "empagliflozin",
                "class": "SGLT2 Inhibitors"
            },
            {
                "brand": "Metformin",
                "generic": "metformin",
                "class": "Biguanides"
            }
        ]

        return suggestions[:max_suggestions]


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_retriever(predefined_answers: Optional[Dict] = None) -> Retriever:
    """
    Create and initialize a Retriever instance.

    Args:
        predefined_answers: Dictionary mapping drug_class to info
            Example: {
                "GLP-1 Receptor Agonists": {
                    "description": "...",
                    "drugs": ["Ozempic", "Wegovy", ...],
                    "category_info": "..."
                }
            }

    Returns:
        Initialized Retriever
    """
    return Retriever(predefined_answers=predefined_answers)


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*60)
    print("RETRIEVER MODULE TEST")
    print("="*60)

    # Example predefined answers
    predefined = {
        "GLP-1 Receptor Agonists": {
            "description": "Medications that mimic GLP-1 hormone to lower blood sugar",
            "drugs": ["Ozempic", "Wegovy", "Victoza", "Trulicity", "Mounjaro"],
            "category_info": "Generally well-tolerated with gastrointestinal side effects"
        }
    }

    # Initialize retriever
    retriever = create_retriever(predefined_answers=predefined)

    print("\n" + "="*60)
    print("TEST 1: Main retrieval (high confidence)")
    print("="*60)

    test_dict_1 = {
        "drug_brand": "Ozempic",
        "drug_generic": "semaglutide",
        "drug_class": "GLP-1 Receptor Agonists",
        "categories": ["adverse_reactions"],
        "confidence": "high"
    }

    result_1 = retriever.retrieve(
        extracted_dict=test_dict_1,
        original_query="What are the side effects of Ozempic?"
    )

    print(f"\nResult type: {result_1.result_type}")
    print(f"Chunks returned: {len(result_1.chunks) if result_1.chunks else 0}")
    if result_1.chunks:
        print(f"Top chunk score: {result_1.chunks[0].similarity_score:.4f}")
        if result_1.chunks[0].rerank_score:
            print(f"Top rerank score: {result_1.chunks[0].rerank_score:.4f}")

    print("\n" + "="*60)
    print("TEST 2: Drug class query")
    print("="*60)

    test_dict_2 = {
        "drug_brand": None,
        "drug_generic": None,
        "drug_class": "GLP-1 Receptor Agonists",
        "categories": ["general"],
        "confidence": "high"
    }

    result_2 = retriever.retrieve(
        extracted_dict=test_dict_2,
        original_query="Tell me about GLP-1 agonists"
    )

    print(f"\nResult type: {result_2.result_type}")
    print(f"Message: {result_2.message[:100]}..." if result_2.message else "No message")

    print("\n" + "="*60)
    print("TEST 3: Low confidence")
    print("="*60)

    test_dict_3 = {
        "drug_brand": "Ozempick",
        "drug_generic": None,
        "drug_class": None,
        "categories": ["warnings"],
        "confidence": "low"
    }

    result_3 = retriever.retrieve(
        extracted_dict=test_dict_3,
        original_query="Is Ozempick safe?"
    )

    print(f"\nResult type: {result_3.result_type}")
    print(f"Awaiting confirmation: {result_3.awaiting_confirmation}")
    print(f"Message: {result_3.message[:100]}..." if result_3.message else "No message")

    print("\n" + "="*60)
    print(f"✓ Tests complete!")
    print(f"Reranker status: {'ON' if RERANKER_ON else 'OFF'}")
    print("="*60)