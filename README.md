# MAM05-project
End-to-end pipeline (runtime flow — concise)

1. User input (free text) → UI_main receives /explain.
2. normalizer.extract_candidates → normalizer.map_to_rxnorm → if low confidence return confirmation step; else continue.

3. For each confirmed RxCUI → cache.get_cached_spl. If miss [NOT IN OUR CASE]:
   * api_clients.openFDAClient.fetch_by_setid → parsers.parse_json → chunker.chunk_text → cache.save_cached_spl.

4. For any new chunks → embedder.embed_texts in batch → vector_index.add.

5. For each RxCUI → retriever.retrieve_for_med (build_structured_query, embed, vector_index.query, re-rank, threshold) → returns top chunks.

6. rules_engine.check_red_flags runs deterministic checks (DrugCentral interactions + keyword scans).

7. prompt_builder.build_prompt assembles evidence + task + few-shot examples.

8. llm_client.call_llm(prompt) invokes UvA LLM → get response.

9. llm_client.validate_response ensures chunk ids are cited; if invalid, fallback: return conservative message "No authoritative evidence found" and flag for clinician review.

10. UI_main returns structured JSON: patient-friendly summary(s), chunk provenance (source URLs + chunk ids), flags and confidence metrics.

11. logging_eval.log_query stores all artifacts for auditing and later evaluation.