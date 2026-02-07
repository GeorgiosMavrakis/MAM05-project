"""
Orchestration for UI

Purpose: orchestrate modules to run an end-to-end request cycle (/explain), handle confirmations.

Input: client JSON {"text":"I take ..."}

Output: structured JSON result to UI:

{
 "normalized":[{"name":"sertraline","rx_cui":"12345","confidence":0.92}],
 "summaries":[{"rx_cui":"12345","summary":"...","sources":[...]}],
 "flags":[{"type":"interaction","message":"..."}],
 "raw_llm":"..."
}


Example: returns the patient-friendly messages, chunk provenance, and flags.

Notes: orchestrates UI confirmation if normalizer confidence low.
"""

# api_server.py (FastAPI)
# POST /explain { "text": "...", "confirm": optional }
# 1. call normalizer.extract_candidates, map_to_rxnorm
# 2. if any low confidence -> return matched list + "please confirm" OR accept user override
# 3. for confirmed rx_cuis: for each med call retriever.retrieve_for_med(...)
# 4. collect chunks, run rules_engine.check_red_flags
# 5. build prompt and call llm_client.call_llm
# 6. validate response, attach flags, return structured JSON:
#    { "summaries": [...], "flags": [...], "sources": [...], "raw_llm": "..." }
