"""
RETRIEVER MODULE — RAG Pipeline
Embeds a user query, retrieves the top-N semantically similar chunks
from Qdrant, reranks them with a cross-encoder, and returns a
structured dictionary ready for downstream LLM consumption.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================
# CONFIGURATION
# ============================================

QDRANT_URL        = "http://134.98.133.84:6333"
QDRANT_API_KEY    = None                          # ⚠️  Set via env var in production
COLLECTION_NAME   = "drug_embeddings"

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"           # Must match the model used at index time
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_TOP_K   = 80    # Candidates pulled from Qdrant via vector similarity
RERANK_TOP_K      = 30    # Final chunks kept after cross-encoder reranking

RERANKER_ON       = True  # If False, skip cross-encoder and keep top RERANK_TOP_K by vector score

# Near-duplicate deduplication threshold (Jaccard word-set similarity).
# Two chunks whose overlap exceeds this are considered duplicates; the
# lower-ranked one is dropped. Tune between 0.0 (drop nothing) and 1.0
# (only drop exact copies). 0.85 removes paraphrased generics while
# keeping genuinely different sections of the same drug label.
DEDUP_THRESHOLD   = 1

QDRANT_TIMEOUT    = 120.0  # seconds — increase if the VM/network is slow
QDRANT_RETRIES    = 3      # number of attempts before giving up

# When True  → each chunk in the output contains only {"id": ..., "text": ...}
# When False → each chunk retains the full Qdrant payload (text + chunk_index + metadata)
PLAIN_CHUNKS      = True

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================
# LAZY-LOADED SINGLETONS
# Avoids re-loading heavy models on every call
# when the module is imported once and reused.
# ============================================

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None        = None
_qdrant:   QdrantClient | None        = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        logger.info("Connecting to Qdrant at %s", QDRANT_URL)
        _qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
        )
    return _qdrant


# ============================================
# HELPERS
# ============================================

def _embed_query(query: str) -> list[float]:
    """
    Encode the user query with the same model used at index time.
    normalize_embeddings=True keeps it consistent with how chunks
    were embedded in embedder.py (normalize_embeddings=True there too).
    """
    embedder = _get_embedder()
    vector = embedder.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def _build_chunk_id(point_id: int | str, payload: dict[str, Any]) -> str:
    """
    Attempt to reconstruct the original string id from metadata fields.
    Falls back to the Qdrant integer point id if metadata is missing/incomplete.

    Original id format: "{set_id}_{category}_{chunk_index}"
    e.g.  "5f76e16f-0c24-454a-81af-4080b312c940_spl_product_data_elements_0"
    """
    metadata = payload.get("metadata") or {}
    set_id   = metadata.get("set_id")
    category = metadata.get("category")
    chunk_idx = metadata.get("chunk_index", payload.get("chunk_index"))

    if set_id and category and chunk_idx is not None:
        return f"{set_id}_{category}_{chunk_idx}"

    # Graceful fallback — guarantees a non-empty id for citation
    return str(point_id)


def _format_chunks(
    points: list[Any],
    plain: bool = PLAIN_CHUNKS,
) -> list[dict[str, Any]]:
    """
    Convert Qdrant ScoredPoint objects into the output chunk format.

    plain=True  → {"id": ..., "text": ...}
    plain=False → {"id": ..., "text": ..., "chunk_index": ..., "metadata": ..., "score": ...}
    """
    chunks = []
    for point in points:
        payload  = point.payload or {}
        chunk_id = _build_chunk_id(point.id, payload)
        text     = payload.get("text", "")

        if plain:
            chunks.append({"id": chunk_id, "text": text})
        else:
            chunks.append({
                "id":          chunk_id,
                "text":        text,
                "chunk_index": payload.get("chunk_index"),
                "metadata":    payload.get("metadata", {}),
                "score":       point.score,
            })

    return chunks


# ============================================
# CORE STEPS
# ============================================

def _retrieve_from_qdrant(
    query_vector: list[float],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[Any]:
    """
    Query Qdrant using the new query_points() API (replaces deprecated .search()).

    The `query` parameter is wrapped in `models.Query` with an explicit dense
    vector — this resolves the IDE type warning that arises from incomplete stubs
    which only expose the scalar (int/str/UUID) overloads of the parameter.

    Retries up to QDRANT_RETRIES times on ReadTimeout, which can happen on
    the first cold call to a remote VM over WAN.

    Returns a list of ScoredPoint objects, already sorted by descending score.
    """
    import time
    client = _get_qdrant()

    last_exc: Exception | None = None
    for attempt in range(1, QDRANT_RETRIES + 1):
        try:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                # Pass the raw float list directly — this is correct at runtime.
                # models.Query is a typing.Union alias, NOT an instantiable class,
                # so models.Query(nearest=...) raises "Cannot instantiate typing.Union".
                # The IDE warning ("Expected int|str|UUID|PointId") is a stub gap;
                # list[float] is a valid dense-vector query per the Qdrant docs.
                query=query_vector,  # type: ignore[arg-type]
                limit=top_k,
                with_payload=True,   # we need text (and optionally metadata) back
                with_vectors=False,  # skip the raw 384-dim embedding — saves bandwidth
            )
            return response.points

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Qdrant query attempt %d/%d failed: %s",
                attempt, QDRANT_RETRIES, exc,
            )
            if attempt < QDRANT_RETRIES:
                wait = 2 ** attempt  # exponential back-off: 2s, 4s, ...
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)

    raise RuntimeError(
        f"Qdrant query failed after {QDRANT_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


def _rerank(
    query: str,
    points: list[Any],
    top_k: int = RERANK_TOP_K,
) -> list[Any]:
    """
    Vanilla cross-encoder reranker.

    Scores each (query, chunk_text) pair with a cross-encoder and
    returns the top_k points sorted by descending reranker score.
    The cross-encoder sees both query and passage jointly, giving
    much better relevance signal than the bi-encoder similarity alone.
    """
    reranker = _get_reranker()

    texts = [point.payload.get("text", "") for point in points]
    pairs = [(query, text) for text in texts]

    # predict() may return shape (n,) or (n,1) depending on the cross-encoder
    # version. Flattening guarantees we always get a 1D array of scalars.
    # Without this, zip(scores, points) pairs numpy sub-arrays with points;
    # Python's sort then receives a numpy boolean array when comparing two
    # scores (instead of a scalar bool), making the order undefined and causing
    # the same point to appear repeatedly in the output.
    scores = reranker.predict(pairs, show_progress_bar=False).flatten()

    # Index-based sort avoids any secondary comparison of ScoredPoint objects
    # when two scores happen to be exactly equal (which would also crash sort).
    sorted_indices = sorted(range(len(points)), key=lambda i: scores[i], reverse=True)

    return [points[i] for i in sorted_indices[:top_k]]


# ============================================
# DEDUPLICATION
# ============================================

def _jaccard(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity on word sets: |A ∩ B| / |A ∪ B|.

    Chosen over embedding similarity because:
    - Zero extra model calls — purely lexical, runs in microseconds.
    - Drug label near-duplicates differ only in punctuation/formatting,
      so word overlap is a precise and reliable signal.
    - A threshold of ~0.85 comfortably separates true duplicates (same
      contraindication text from different generic manufacturers, Jaccard
      typically 0.90-0.99) from legitimately distinct sections of the
      same label (typically < 0.60).
    """
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _deduplicate(points: list[Any], threshold: float = DEDUP_THRESHOLD) -> list[Any]:
    """
    Greedy near-duplicate removal.

    Iterates through points in ranked order (best first). For each candidate,
    computes Jaccard against every already-accepted chunk. If similarity
    exceeds `threshold` with ANY accepted chunk, the candidate is dropped.

    O(n²) but n ≤ 70 here, so negligible in practice.
    """
    accepted: list[Any] = []
    accepted_texts: list[str] = []

    for point in points:
        text = (point.payload or {}).get("text", "")
        is_duplicate = any(
            _jaccard(text, seen) >= threshold
            for seen in accepted_texts
        )
        if not is_duplicate:
            accepted.append(point)
            accepted_texts.append(text)

    dropped = len(points) - len(accepted)
    if dropped:
        logger.info(
            "  → %d near-duplicate chunk(s) removed (Jaccard threshold=%.2f)",
            dropped, threshold
        )

    return accepted


# ============================================
# PUBLIC API
# ============================================

def retrieve(user_query: str) -> dict[str, Any]:
    """
    Full RAG retrieval pipeline.

    Steps
    -----
    1. Embed the user query with all-MiniLM-L6-v2.
    2. Retrieve the top-80 semantically similar chunks from Qdrant
       using the new query_points() API.
    3. Deduplicate near-identical chunks immediately via Jaccard
       word-set similarity — so the reranker only scores distinct content.
    4a. If RERANKER_ON: rerank deduplicated candidates with cross-encoder,
        keeping top RERANK_TOP_K.
    4b. If RERANKER_ON is False: keep top RERANK_TOP_K by vector score.
    5. Format and return results.

    Parameters
    ----------
    user_query : str
        The raw user question / prompt.

    Returns
    -------
    dict with keys:
        "user_prompt"  : str         — the original query, unchanged
        "text_chunks"  : list[dict]  — retrieved & reranked chunks
                         Each dict has at minimum "id" and "text".
                         If PLAIN_CHUNKS=False, also "chunk_index",
                         "metadata", and "score" are included.
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")

    logger.info("Query: %r", user_query)

    # ── Step 1: embed ──────────────────────────────────────────────
    logger.info("Embedding query...")
    query_vector = _embed_query(user_query)

    # ── Step 2: vector search in Qdrant ───────────────────────────
    logger.info("Retrieving top-%d candidates from Qdrant...", RETRIEVAL_TOP_K)
    candidates = _retrieve_from_qdrant(query_vector, top_k=RETRIEVAL_TOP_K)
    logger.info("  → %d candidates returned", len(candidates))

    if not candidates:
        logger.warning("Qdrant returned 0 results. Check collection name and connectivity.")
        return {"user_prompt": user_query, "text_chunks": []}

    # ── Step 3: near-duplicate removal (before reranking) ─────────
    # Deduplicating here means the reranker only scores genuinely distinct
    # chunks — it won't waste capacity ranking near-identical generic copies.
    logger.info("Deduplicating (Jaccard threshold=%.2f)...", DEDUP_THRESHOLD)
    candidates = _deduplicate(candidates, threshold=DEDUP_THRESHOLD)
    logger.info("  → %d unique candidates after deduplication", len(candidates))

    # ── Step 4: reranking (conditional) ───────────────────────────
    if RERANKER_ON:
        logger.info("Reranking to top-%d with cross-encoder...", RERANK_TOP_K)
        selected = _rerank(user_query, candidates, top_k=RERANK_TOP_K)
        logger.info("  → %d chunks after reranking", len(selected))
    else:
        logger.info("Reranker OFF — keeping top-%d by vector score", RERANK_TOP_K)
        selected = candidates[:RERANK_TOP_K]

    # ── Step 5: format output ──────────────────────────────────────
    text_chunks = _format_chunks(selected, plain=PLAIN_CHUNKS)

    return {
        "user_prompt": user_query,
        "text_chunks": text_chunks,
    }


# ============================================
# QUICK SANITY CHECK (run as script)
# ============================================

if __name__ == "__main__":
    import json

    test_query = "What are the contraindications for metformin?"
    print("=" * 60)
    print("RETRIEVER — QUICK TEST")
    print("=" * 60)
    print(f"\nQuery: {test_query}\n")

    result = retrieve(test_query)

    print(f"user_prompt : {result['user_prompt']}")
    print(f"text_chunks : {len(result['text_chunks'])} chunks returned\n")

    print("── First chunks ──")
    for chunk in result["text_chunks"][:29]:
        print(json.dumps(chunk, indent=2, ensure_ascii=False))
        print()
