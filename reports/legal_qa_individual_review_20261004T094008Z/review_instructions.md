# Individual legal QA candidate review

Open one JSON file in `candidate_forms` at a time. Each file shows the question, proposed answer, complete supporting passage, source/page, citation metadata, and duplicate/template warnings. Edit only its `review` object; candidate details identify the immutable Stage 15 record.

Set `decision` to `APPROVE`, `REJECT`, `EDIT_AND_RECHECK`, or `PENDING`. Every completed decision requires a reviewer identifier and a concise reason. For `APPROVE`, set both confirmation fields to `YES` only after personally checking that the answer is supported by the passage and checking the legal reference against the original legal source. Unconfirmed references and uncertain interpretations remain pending until that source check is complete. Automated page matches and scores are not legal verification.

`EDIT_AND_RECHECK` requires `edited_answer`; it does not approve the candidate. Submit a later review form with an explicit `APPROVE` only after reviewing the edit. Do not change candidate fields or silently correct citations. When later approving a changed answer, set `edited_answer_rechecked_confirmation` to `YES` only after rechecking it. A revised citation may be entered as `reviewed_legal_reference` only after checking the original source; it must also pass the deterministic evidence check.

To validate/import completed forms into a separate audit output, run:

```powershell
& .\venv\Scripts\python.exe scripts\review_first_batch_legal_qa.py import --forms-dir <this-output>\candidate_forms --batch-csv <Stage-15-first_review_batch.csv> --output-dir <new-empty-output-directory>
```

Only explicit approvals with both confirmations and passing deterministic provenance/evidence checks are emitted to the separate approved-review file. That file is not a train/validation/test split and does not establish legal correctness by itself.
