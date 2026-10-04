# Interactive legal QA review session

Run the session from the repository root in Git Bash. To review the first candidate, use:

```bash
cd /d/Legal_AI_Projects/legal_ai_rag
./venv/Scripts/python.exe scripts/interactive_legal_qa_review.py --start-at 1
```

Each candidate is displayed individually. The helper writes submitted decisions under `decision_forms/` in this new session directory; it does not edit the original Stage 16 forms. The session output path is `reports\legal_qa_review_session_20261004T095310Z`. It also contains a summary after a session completes.

To import and validate decisions later, use the existing importer with a new, unused output directory:

```bash
cd /d/Legal_AI_Projects/legal_ai_rag
./venv/Scripts/python.exe scripts/review_first_batch_legal_qa.py import --forms-dir reports\legal_qa_review_session_20261004T095310Z\decision_forms --batch-csv reports\legal_qa_review_prioritization_20261004T093327Z\first_review_batch.csv --output-dir reports/legal_qa_review_import_<new-unique-name>
```

The importer remains authoritative: it validates explicit decisions and source/evidence constraints. Automated checks do not establish legal correctness, and no dataset split is created by this workflow.
