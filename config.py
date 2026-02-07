"""
Central configuration

Purpose: single source of truth for file paths, API keys, thresholds, model names, and default params.

Input: none at runtime (read constants / environment variables).

Output: variables used by other modules (strings, numbers).

Example:
"""

DAILYMED_DUMP_PATH = "data/dailymed/"
DRUGCENTRAL_PATH = "data/drugcentral/"
# RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_DIR = "vectors/"
LLM_API_KEY = "<from env>"
EMBED_BATCH = 128
RETRIEVE_K_POOL = 8
RETRIEVE_K_FINAL = 3
SIM_THRESHOLD = 0.65
