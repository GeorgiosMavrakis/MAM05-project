"""
Extract & canonicalize mentions + extract concerns

Purpose: turn free-text user input to a small set of canonical medication IDs and normalized concern tokens.

Input: user_text: str

Output: {"candidates": [{"text":"sertraline","rx_cui":"12345","confidence":0.92}, ...],
"concerns":[{"label":"dizziness","confidence":0.87}]}

Example: user "I take sertraline 50 mg and sometimes ibuprofen. Lately dizzy." → output shown above.

Notes: internally may call an LLM extractor OR a fuzzy matcher + RxNavClient.approx_match. If confidence low,
API returns ambiguous=True so UI can ask user to confirm.
"""
from typing import List

# normalizer.py
def extract_candidates(user_text: str) -> List[str]:
    # Option A: LLM extractor -> short list of candidate strings
    # Option B: fast fuzzy token matching against alias CSV
    return ["sertraline", "ibuprofen"]

def map_to_rxnorm(candidates: List[str]) -> List[{"name":str,"rx_cui":str,"confidence":float}]:
    # call RxNav approximateTerm.json:
    # requests.get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json", params={"term": drug_name})
    # return candidate rxcuids sorted by score

    # if confidence < threshold mark as ambiguous
    return [{"name":"sertraline","rx_cui":"12345","confidence":0.92}, ...]

def extract_concerns(user_text: str) -> List[{"label":str,"confidence":float}]:
    # Option: call LLM few-shot or use embedding-match to small canonical list
    return [{"label":"dizziness","confidence":0.87}]
