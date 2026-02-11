"""
Call the LLM and return raw model output

Purpose: send the prompt to UvA LLM (or other) and receive response; handle retries, rate limits.

Input: prompt_text (str)

Output: {"text": "<model output>", "tokens": 420, "meta": {...}}

Example: returns text that includes a list of chunk ids and a short patient-friendly paragraph.

Notes: append received tokens for logging; perform basic sanitization.
"""
from typing import List
import openai
import database
import requests

def ask_rag(question: str):
    """
    Query the RAG system with a question.
    Returns the LLM response as a string.
    """
    try:
        docs = database.search_db(question)
        context = "\n\n".join(docs)

        prompt = f"""Use only the context to answer.

Context:
{context}

Question:
{question}

Answer:
"""

        endpoint = "https://ai-research-proxy.azurewebsites.net/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer YOUR_API_KEY_HERE"  # TODO Add API key
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        res = requests.post(endpoint, json=payload, headers=headers)
        res.raise_for_status()
        data = res.json()

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"Error calling LLM: {str(e)}"
    except KeyError:
        return "Error: Unexpected response format from LLM"
