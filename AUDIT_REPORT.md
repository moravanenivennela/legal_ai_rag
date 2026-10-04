# Legal AI RAG project audit — source review

## Scope
Reviewed the source archive, application, RAG engine, ingestion pipeline, fine-tuning scripts, requirements, and saved evaluation artifacts. Python syntax compilation passed in the review environment. The excluded local Chroma index, Ollama model files, and fine-tuning adapter weights were not available in this archive, so the full app and end-to-end model tests could not be run.

## Fixes applied in this audit copy

1. Expanded `requirements.txt` to include packages imported by the Streamlit app, RAG pipeline, image model, LoRA scripts, and evaluation utilities.
2. Updated the app description to remove the stale “no fine-tuning” and unqualified benchmark-improvement claims, and label the displayed metric table as historical experimental results.
3. Updated `LegalRAGEngine.sparse_search` to rank by document index instead of `list.index(doc)`, which can attach incorrect metadata when duplicate chunks exist.
4. Improved exact provision search to ignore common contents/schedule/appendix entries, extract a bounded provision excerpt, and return the first non-contents match rather than every repeated numbered reference.
5. Routed explicit `Section N` questions to the indexed Consumer Protection Act after out-of-domain law checks, and allowed a found exact provision to bypass the dense-distance guardrail.
6. Reworked `fine_tuning/scripts/train_lora.py` to use the tokenizer's chat template, remove the duplicated assistant-start token, and mask prompt tokens so loss is computed only on target-answer tokens.
7. Removed the misleading `accuracy` field derived from a fixed token-F1 threshold in the RAG engine metric methods, and updated the two model-comparison scripts plus QA comparison to call it an overlap-threshold pass rate instead.

## Important issues still requiring data/model-backed testing

- **End-to-end startup is unverified.** `chroma_db/`, Ollama models, and adapter weights are local artifacts excluded from the source bundle. Rebuild the index with `python ingest.py`, confirm Ollama has the selected model, then launch the app.
- **Legal QA accuracy is not established by token F1 or source match.** The saved fine-tuning evaluation contains only 21 test examples and reports 0% exact match; token F1 is text overlap only. Human review against official legal text is still needed.
- **Groundedness is a proxy.** NLI entailment probabilities do not prove every legal statement is correct. Do not present the score as verified legal accuracy.
- **Lexical overlap remains a proxy.** The engine now names the F1-threshold result `overlap_threshold_pass`; token precision/recall/F1 against context or reference text are not legal correctness or semantic factuality. Older saved CSV/JSON/plots may still contain historical columns labeled `accuracy` and should not be re-used as current validated results.
- **Dataset scale is small for fine-tuning.** The original QA set has 246 examples (198 train, 27 validation, 21 test); the approved set has 197 train and 27 validation, with a separate 21-question test file. More reviewed, diverse examples are needed before making performance claims.
- **Metadata quality remains a risk.** Ingestion assigns page-level citation metadata based on the first reference detected on the page; chunks on a page can discuss different provisions. Citation correctness should be tested against the exact source passage.
- **`ingest.py` deletes and rebuilds the Chroma collection.** Back up the existing index before rerunning ingestion if you need to preserve it.
- **`trust_remote_code=True` is used in fine-tuning/evaluation scripts.** Only use it with a model repository you trust.
- **Legal safety:** Keep a visible limitation that this is an informational research prototype, not legal advice, and direct users to verify provisions in official sources.

## Test status

- Python source syntax compilation: passed for the extracted source tree.
- Retrieval/metric regression tests: 6/6 passed for exact Article/Section filtering, Section routing, duplicate-text metadata alignment, and exact-hit guardrail bypass. Run with `python tests/test_retrieval_logic.py`.
- JSONL parsing and basic train/validation/test overlap checks: passed for the inspected splits; no exact question-string overlap was found between the `train.jsonl`, `validation.jsonl`, `test.jsonl` splits or between `qa_train.jsonl`, `qa_validation.jsonl`, `qa_test.jsonl`.
- Dataset quality spot-check: required fields were present in inspected QA splits, and no exact question-string overlap was found across train/validation/test; 9 train/validation records were flagged for low lexical overlap between answer and evidence, which needs manual semantic review.
- Model generation, embedding retrieval against the real index, LoRA training, and UI startup: not run because model/index artifacts and runtime dependencies are not fully available in this review environment.
