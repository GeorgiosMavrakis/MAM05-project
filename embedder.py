"""
Produce numeric embeddings for texts

Purpose: convert chunk texts (and query strings) into numeric vectors.

Input: list of strings ["text1", "text2", ..."]

Output: list of vectors [[0.12, ...], [0.23, ...]] (floats)

Example: embed_texts(["dizziness reported in trials"]) → [[0.013, -0.22, ...]]

Notes: batch embedding offline for entire corpus; use same model for query embedding.
"""
from typing import List
from sentence_transformers import SentenceTransformer
import config

model = SentenceTransformer(config.EMBED_MODEL)

def embed_texts(texts: List[str]) -> List[List[float]]:
    # batch model.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=True)
    pass
