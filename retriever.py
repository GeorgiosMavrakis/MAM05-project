"""
Build structured query, call vector index, re-rank & filter

Purpose: build the query embedding and return the top-K trustworthy chunks for a med+concern.

Input: rx_cui, drug_name, concern (strings) and optional sections list.

Output: [{chunk_id, text, meta, score}, ...] (filtered, re-ranked, ≤ K_final)

Example: input ("12345", "sertraline", "dizziness") → returns 3 chunk dicts focused on dizziness/adverse reactions.

Notes: for downloaded DB use filter_meta={"rx_cui":"12345"} to ensure drug-level retrieval.
"""
from typing import List
import config, embedder, vector_index

# retriever.py
def build_structured_query(rx_cui: str, drug_name:str, concern:str, sections:List[str]) -> str:
    return f"RxCUI:{rx_cui} {drug_name} | concern:{concern} | sections:{','.join(sections)}"

def retrieve_for_med(rx_cui, drug_name, concern) -> List[chunks]:
    q = build_structured_query(rx_cui, drug_name, concern, ["AdverseReactions","DrugInteractions"])
    q_vec = embedder.embed_texts([q])[0]
    candidates = vector_index.query(q_vec, filter_meta={"rx_cui": rx_cui}, k=config.RETRIEVE_K_POOL)
    # re-rank by cosine(q_vec, candidate_vec) and apply SIM_THRESHOLD
    return candidates[:config.RETRIEVE_K_FINAL]