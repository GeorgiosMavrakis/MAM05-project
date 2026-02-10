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
import config, chunker, database
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> List[float]:
    vector = embedder.encode(text)

    return vector.tolist()

def embedder():

    with open(DATA_FILE, "r", encoding="utf-8") as f: # TODO Add file path
        text = f.read()

    chunks = chunker.chunk_text(text)

    ids = []
    docs = []
    embeds = []

    for i, chunk in enumerate(chunks):

        ids.append(str(i))
        docs.append(chunk)
        embeds.append(get_embedding(chunk))

    database.collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeds
    )
    database.chroma_client.persist()

