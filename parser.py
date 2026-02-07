"""
Parse raw dataset documents into plain text sections

Purpose: extract the structured sections you care about from raw source content (XML/CSV/HTML).

Input: raw data from api_clients (e.g., SPL XML string) or file path for downloaded files.

Output: list of dictionaries: [{ "section":"Adverse Reactions", "text":"..cleaned plain text..", "source_url":"...",
"date":"2024-11-01" }, ...]

Example: parse SPL XML → [{section:"Contraindications", text:"Do not use in patients with X", ...}]

Notes: For downloaded dumps you parse local files; for live API you parse response body.
Keep parser logic identical for both.
"""
from typing import List

# parsers.py
def parse_json(json_text: str) -> List[{"section":"AdverseReactions","text": "..."}]:
    pass
