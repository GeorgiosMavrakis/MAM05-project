"""
Break long sections into retrievable chunks

Purpose: create coherent text chunks (150–400 words) with metadata for indexing.

Input: parser output (list of section dicts) + rx_cui metadata.

Output: list of chunk dicts: [{ "chunk_id":"12345-ADR-0", "rx_cui":"12345", "section":"Adverse Reactions",
"text":"...", "source":"openFDA", "date":"...", }, ...]

Example: one Adverse Reactions section → 2 chunks in list.

Notes: keep original section and chunk summary linkable.
"""
from typing import List


# chunker.py
def chunk_text(text: str) -> List[str]:

    chunks = []
    start = 0

    while start < len(text):
        end = start + 500 # 500 chunk size
        chunks.append(text[start:end])
        start = end - 50 # 50 chunk overlap

    return chunks
