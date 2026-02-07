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
def chunk_text(text: str, rx_cui: str, section: str) -> List[{"chunk_id":str,"text":str,"rx_cui":str,"section":str}]:
    # split on sentence boundaries, accumulate ~200 words per chunk
    # generate chunk_id e.g., f"{rx_cui}-{section}-{i}"
    pass
