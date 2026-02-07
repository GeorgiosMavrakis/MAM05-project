"""
Create a strict evidence block + task prompt

Purpose: format retrieved chunks and task instructions into a prompt that forces the LLM to cite sources.

Input: retrieved chunks (list), patient_concern string, few-shot examples (optional), reading level param.

Output: prompt_text: str ready to send to the LLM.

Example: returns a prompt beginning with system instruction "Use ONLY the evidence blocks below..." then
CHUNK #1 - DailyMed - Adverse Reactions - <text> etc., then task instructions: 1) list the chunk ids that support
each claim 2) produce ≤120-word patient summary.

Notes: include chunk ids and source URLs in evidence block.
"""
from typing import List

# prompt_builder.py
def build_prompt(chunks: List, patient_concern: str, reading_level="simple") -> str:
    # SYSTEM instruction: "You MUST use only the evidence below; if unsupported say 'No authoritative evidence found.'"
    # EVIDENCE blocks: CHUNK # - SRC - SECTION - TEXT
    # TASK: 1) list chunk ids supporting each claim 2) produce <=120-word patient summary with bullets 3) list sources
    return prompt_text
