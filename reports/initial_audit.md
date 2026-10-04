# Initial read-only audit

**Audit date:** 2026-10-04  
**Repository:** `legal_ai_rag`  
**Mode:** Read-only inspection. No application code, datasets, models, checkpoints, or evaluation artifacts were changed. No tests, training, model downloads, application startup, or ingestion were run.

## Executive summary

The project has the main pieces of a legal RAG system, but the current evidence does **not** show that it reliably produces correct legal answers. The strongest risks are:

1. The current LoRA training script refers to dataset variables that it never creates, so training will stop before it starts.
2. The Streamlit app sends generation requests to Ollama; it does not load the Qwen LoRA adapter used by the fine-tuning evaluation. The repository's root `Modelfile` describes a Llama 3.2 model, not the Qwen base model used for the adapters.
3. The app generates an answer even when retrieval reports failure and provides no context. Uploaded-document retrieval also has no minimum relevance threshold.
4. The saved “V3” 80% metric is from 30 domain-classification examples, not a 500-question retrieval or answer-correctness evaluation. The archived 30-answer report shows 100% source match but only 45.02% average automated groundedness.
5. Citation text is requested from the LLM but is not checked. Ingestion stores one provision label per PDF page, then reuses it for every chunk on that page.
6. Re-running ingestion deletes and recreates the existing Chroma collection. Do not run it against the only copy of the index.

There is some positive leakage evidence: the reviewed primary passage and QA splits had no exact passage/question overlap. This is not proof that every candidate dataset is free of semantic duplicates or leakage.

## Prioritized findings

### P0 — Protect the existing retrieval index before any ingestion run

[`ingest.py`](../ingest.py#L88) deletes the existing `legal_normative_docs` Chroma collection before creating a replacement. It then writes the BM25 index. A failed or interrupted rebuild can leave the app without its previous working index; no backup or atomic swap is present in this script. **This audit did not run ingestion.** Back up the full `chroma_db` directory before any future rebuild, and make rebuilding recoverable before treating it as routine.

### P0 — The current LoRA trainer has an undefined-variable stop

[`train_lora.py`](../fine_tuning/scripts/train_lora.py#L158) creates `train_dataset` and `validation_dataset`, but [`Trainer`](../fine_tuning/scripts/train_lora.py#L284) is given `tokenized_train` and `tokenized_validation`. Those names are not defined elsewhere in the file. If setup reaches Trainer construction, Python will raise `NameError` before training starts.

This means the existing adapter files and saved evaluation results are evidence of prior artifacts/runs, not evidence that the current on-disk trainer works. The current training settings use Qwen2.5-0.5B-Instruct, one epoch, and 197 approved training / 27 validation records. The saved approved-adapter evaluation has 21 examples, 0% exact match, and average token F1 0.6197; token F1 is text overlap, not legal correctness. See [`train_lora.py`](../fine_tuning/scripts/train_lora.py#L24), [`evaluation_approved_finetuned.json`](../fine_tuning/evaluation_approved_finetuned.json#L1), and [`evaluate_finetuned.py`](../fine_tuning/scripts/evaluate_finetuned.py#L16).

### P1 — The app does not load the Qwen adapter for inference

The app offers Ollama tags (`legal-ai-finetuned`, `llama3.2:1b`, `llama3.2:3b`) and passes the selected name to `LegalRAGEngine`; generation calls `ollama.chat`. There is no `PeftModel` or adapter-loading path in app inference. The separate fine-tuning evaluator does call `PeftModel.from_pretrained`, but that proves only that its offline evaluator loads the adapter.

The saved adapter configuration identifies Qwen2.5-0.5B-Instruct as its base. In contrast, the repository's [`Modelfile`](../Modelfile#L2) starts from `Llama-3.2-3B-Instruct.Q4_K_M.gguf` and has no adapter directive. Therefore the project code/configuration does not wire the evaluated Qwen adapter into app inference. The exact contents of a model tag in a separately running Ollama service were not inspected.

### P1 — The app can generate without trustworthy retrieved context

[`app.py`](../app.py#L346) receives a retrieval-confidence flag, but does not use it to decide whether to answer. The call to `generate_stream` still happens after the optional `if contexts` display block, including when `contexts` is empty ([`app.py`](../app.py#L352), [`app.py`](../app.py#L378)). The prompt asks the model not to hallucinate, but this is not an enforcement mechanism.

Uploaded-document retrieval selects the top two chunks by cosine similarity but has no minimum score cutoff ([`rag_engine.py`](../rag_engine.py#L541)). Any returned upload chunk sets the confidence flag to true in the app, even if it is a poor match ([`app.py`](../app.py#L347)). This can let an unrelated chunk be treated as support. The app should abstain when retrieval is inadequate and apply a tested similarity threshold to uploads.

### P1 — “500-question evaluation” does not establish complete answer correctness

There are three different evaluation paths; their metrics must not be presented as interchangeable:

| Artifact / script | What it actually measures |
|---|---|
| [`evaluate_rag_retrieval.py`](../evaluate_rag_retrieval.py#L1) | Retrieval success and Hit@k/MRR against an expected source. It does not evaluate the generated answer. |
| [`evaluate_rag_answers_500.py`](../evaluate_rag_answers_500.py#L513) | Runs retrieval, Ollama generation, and automated NLI scoring for 500 hard-coded candidate questions. It records source match and groundedness, but has no gold/reference answer or human correctness score. Its own header calls the questions synthetic candidates. |
| [`run_fresh_v3_eval.py`](../run_fresh_v3_eval.py#L49) | Predicts domain labels with the raw classifier. It does not run retrieval or generation. |

The current [`outputs/RAG_RETRIEVAL_EVALUATION_500_v3.csv`](../outputs/RAG_RETRIEVAL_EVALUATION_500_v3.csv) has **30 data rows**, 15 per in-domain statute. The saved [`evaluation_metrics_v3.txt`](../outputs/evaluation_metrics_v3.txt) reports 80% accuracy and zero out-of-domain support. It is therefore a 30-row, two-class domain-label result, not a 500-question end-to-end answer result.

The generic [`evaluate_rag_answers.py`](../evaluate_rag_answers.py#L59) and [`evaluate_rag_retrieval.py`](../evaluate_rag_retrieval.py#L61) evaluators default to `outputs/rag_evaluation_questions_500.csv`, which is currently absent from the working tree. Their loaders fall back to a legacy 30-question list when that file is missing; the report can still be written to a filename containing `500`. The archived [`outputs.zip`](../outputs.zip) contains the older 30-row answer and retrieval reports, not a saved 500-row answer-correctness result.

The archived 30-answer CSV/report records 30/30 retrieval successes and source matches, but average automated groundedness is only **45.02%** and only **8/30** answers score at least 70%. One saved answer to “What does Article 14 guarantee?” says it guarantees the right to life and liberty, which appears to confuse Article 14 with Article 21. These results show why source match cannot stand in for answer correctness.

### P1 — Citations are not validated and can point to the wrong provision

[`ingest.py`](../ingest.py#L29) takes the first Article/Section/Chapter-like reference found on a PDF page and stores it as that page's citation. All chunks from that page inherit the same metadata ([`ingest.py`](../ingest.py#L77)); a page containing multiple provisions can therefore give a chunk a misleading provision label.

The prompt asks the LLM to cite legal provisions ([`rag_engine.py`](../rag_engine.py#L625)), and the UI displays stored citation metadata, but there is no check that a cited Article/Section exists in the cited chunk or that the answer's claim is supported by it. The NLI score checks answer/context text pairs; it does not validate citations.

### P1 — Ingestion and evaluation have reproducibility / preservation problems

- [`run_fresh_v3_eval.py`](../run_fresh_v3_eval.py#L11) searches several directories for CSVs, then iterates over `set(found_files)` and chooses the first filename matching broad substrings ([line 18](../run_fresh_v3_eval.py#L18)). Which dataset is selected can vary, and it is not fixed by an explicit command-line path.
- The same script predicts with `engine.query_classifier` directly; it does not evaluate the full `classify_query_domain` rule path used by normal retrieval.
- [`requirements.txt`](../requirements.txt) lists packages without pinned versions. The archived blind-evaluation log warns that a classifier pickle trained with scikit-learn 1.9.0 is being loaded under 1.8.0. That can affect compatibility or results.
- At audit start, `git status` showed multiple pre-existing edits and deletions, including core Python files and historical `outputs/` CSVs/reports. Some historical artifacts are present in [`outputs.zip`](../outputs.zip). They were not restored or modified. Review and preserve the existing working-tree changes before any cleanup or commit.

## Metrics and model versions: what can safely be concluded

- [`evaluation_report.txt`](../evaluation_report.txt) gives 86.7% query-domain-classifier accuracy and 100% guardrail accuracy on 16 examples. These are small classification/guardrail checks, not answer-quality measurements.
- The archived blind report gives 99% classification accuracy on 100 domain-labelled questions (30 Constitution, 30 Consumer Protection, 40 out-of-domain). [`final_blind_evaluation.py`](../final_blind_evaluation.py#L342) calls `retrieve` and records the predicted domain; it does not generate or grade legal answers.
- [`classifier_v3_candidate_report.json`](../fine_tuning/classifier_v3_candidate_report.json#L1) describes a different classifier version: 713 train / 99 validation / 50 test records. Its test accuracy is 94%, but its note says synthetic OOD training candidates are not all human-verified. Validation accuracy is 81.8%, with only 30.8% recall on Consumer Protection in that report. These results should not be compared as if they came from the same model and test.
- [`evaluation_finetuned.json`](../fine_tuning/evaluation_finetuned.json#L1) is a five-example adapter run; approved and large-adapter JSONs are separate 21-example runs. Their token F1 values are lexical overlap metrics, not correctness.

## Dataset counts and leakage checks

The following counts were read from the current JSONL/metadata files; no dataset-generation script was run:

| Data | Current count | Audit observation |
|---|---:|---|
| Passage splits | 682 train / 85 validation / 86 test = 853 | Matches [`dataset_metadata.json`](../fine_tuning/dataset/dataset_metadata.json); exact passage text overlap across the three files was zero. |
| Earlier QA split | 20 train / 5 validation / 5 test = 30 | Matches [`qa_generation_metadata.json`](../fine_tuning/dataset/qa_generation_metadata.json); this is an older, smaller dataset. |
| Approved QA | 197 train / 27 validation; separate [`qa_test.jsonl`](../fine_tuning/dataset/qa_test.jsonl) has 21 | Exact question overlap among [`qa_train.jsonl`](../fine_tuning/dataset/qa_train.jsonl), [`qa_validation.jsonl`](../fine_tuning/dataset/qa_validation.jsonl), and test was zero. Approved train/validation also had zero exact full-input overlap and zero shared source/page/evidence with the 21-row QA test. |
| Repaired classifier V3 | 713 train / 99 validation / 50 test | Exact duplicate text within/across these three files was zero. The training file includes synthetic candidates that are explicitly marked not human-verified. |
| Expanded QA candidates | 526 train candidates / 82 validation candidates | These match the generation summary, but are candidates, not reviewed gold answers. |
| [`legal_qa_candidates_500.jsonl`](../fine_tuning/dataset/legal_qa_candidates_500.jsonl) | 0 records | The filename suggests data that is not present in this file; the separate 500-question runner has a hard-coded list. |

These checks rule out exact duplicate leakage in the reviewed splits, not paraphrase/semantic overlap across every candidate, review backup, and evaluation file. The 500 hard-coded evaluation questions use repeated topic/question templates, so topic balance and near-duplicate weighting still need a deliberate audit before treating them as independent test examples.

## Groundedness and retrieval limitations

[`check_groundedness`](../rag_engine.py#L687) splits answers on periods, scores each resulting sentence against every retrieved chunk, takes the best entailment probability for each sentence, and averages those maxima. It assumes score index 1 is entailment ([line 706](../rag_engine.py#L706)); the code does not verify the loaded model's label order. Sentence splitting can also mishandle abbreviations and decimal numbers. The resulting percentage is an automated proxy, not proof of legal truth, and the app displays it only after the answer has already been generated.

Reciprocal Rank Fusion deduplicates by exact text and retains the first hit's metadata ([`rag_engine.py`](../rag_engine.py#L200)). Identical passage text at different source/page locations can therefore inherit the wrong citation metadata. The dense-distance guardrail is a fixed threshold of 1.35 ([line 15](../rag_engine.py#L15)); no calibration report tying that value to false-accept/false-reject rates was found.

## Tests and audit limits

The visible test suite is one file with six lightweight retrieval/metric tests ([`tests/test_retrieval_logic.py`](../tests/test_retrieval_logic.py#L32)). It stubs the heavy runtime libraries and does not cover real PDF ingestion, Chroma/BM25 integration, empty-context abstention, uploaded-document relevance thresholds, citation correctness, NLI label interpretation, model serving, or LoRA training/adapter loading.

**Tests were not run during this read-only audit.** The root [`AUDIT_REPORT.md`](../AUDIT_REPORT.md) contains earlier claims about test results and changes; those claims were not independently rerun here. This report describes the current files and artifacts observed, including the pre-existing dirty working tree.

## Recommended order of work

1. Back up the current index and evaluation artifacts; do not run ingestion until the rebuild is made recoverable.
2. Fix and test the trainer's dataset variable names; verify one small training step before any full run.
3. Explicitly connect one chosen base model and adapter to inference, or clearly label the app as using a separate Ollama model. Verify the model tag/config in Ollama before making performance claims.
4. Make retrieval failure and low-similarity uploads abstain instead of generating unsupported answers.
5. Attach provision-level provenance to chunks and validate answer citations against retrieved text.
6. Freeze evaluation input paths and versions; fail clearly when a 500-question dataset is missing instead of falling back to 30. Separate retrieval, domain classification, NLI, and human-graded answer correctness in reports.
7. Create a reviewed answer-key/claim-level test set, deduplicate it (including paraphrases and repeated templates), and report human-verified answer and citation accuracy alongside retrieval measures.
8. Add focused tests for the missing runtime behaviors above, pin dependencies/model revisions, and record dataset/model hashes and evaluation settings for each result.
