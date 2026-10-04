# Legal QA candidate review

Review each question and answer against the complete evidence passage and the cited source page. Automated page, reference, schema, and duplicate flags are review aids only; they do not establish legal correctness.

Use `APPROVE`, `EDIT_AND_RECHECK`, `REJECT`, or `NEEDS_MORE_EVIDENCE` in `review_decision`. Enter a reviewer identifier and a concise reason for every completed decision. Put a proposed revision in `reviewed_answer`. An edited answer stays out of the reviewed dataset unless it passes deterministic source/page/reference/evidence checks and receives an explicit approval decision. Duplicate and reference-conflict groups are identified by their group IDs; inspect related candidate IDs together.

Leave undecided records blank. Do not treat an empty decision, page match, heading match, or this worksheet as approval.

After saving the completed CSV, import it with:

```powershell
& .\venv\Scripts\python.exe scripts\legal_qa_human_review.py import --review-csv <path-to-review_worksheet.csv> --output-dir <new-empty-output-directory>
```

The import command creates a separate reviewed dataset containing only explicit approvals that pass deterministic validation. Rejected, edited-pending, uncertain, invalid, and undecided rows remain excluded.
