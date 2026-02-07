"""
Deterministic safety checks and interaction lookup

Purpose: apply hard rules independent of LLM: keyword red-flags and authoritative interaction table lookups.

Input: retrieved_chunks, list_of_rx_cuis (from normalizer), optional DrugCentral_interactions_table.

Output: flags = [{"type":"interaction","severity":"major","message":"Increased bleeding risk with
sertraline + ibuprofen", "evidence": [chunk_ids or table rows]}, ...]

Example: finds "bleeding" in Interaction chunk and returns a major flag.

Notes: always run and surface flags even if the LLM does not mention them.
"""
from typing import List

# rules_engine.py
def check_red_flags(retrieved_chunks) -> List[{"type":"contraindication","message":str,"source":...}]:
    # keyword scan: "contraindicat", "anaphylax", "do not use"
    # use DrugCentral interaction rows for pairwise checks -> severity tags
    pass
