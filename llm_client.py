"""
Call the LLM and return raw model output

Purpose: send the prompt to UvA LLM (or other) and receive response; handle retries, rate limits.

Input: prompt_text (str)

Output: {"text": "<model output>", "tokens": 420, "meta": {...}}

Example: returns text that includes a list of chunk ids and a short patient-friendly paragraph.

Notes: append received tokens for logging; perform basic sanitization.
"""
from typing import List

# llm_client.py
def call_llm(prompt: str) -> {"text": str, "tokens": int}:
    # call API using LLM_API_KEY, handle rate limits/backoff
    # return raw response
    pass

def validate_response(response_text: str, required_chunk_ids: List[str]) -> {"ok":bool,"issues":List[str]}:
    # ensure chunk ids referenced, no unsupported factual claims (best-effort)
    pass
