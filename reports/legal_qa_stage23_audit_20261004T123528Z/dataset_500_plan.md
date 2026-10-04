# Scalable plan for a reliable 500-example legal QA set

## Current pool is a queue, not a dataset

The Stage 13 run generated 2,424 raw candidates, kept 1,000 selected rows after a 1,000-candidate cap (the deduplicated pool was 2,113), and labeled selected records pending. The selected source mix is 813 Constitution (81.3%) and 187 Consumer Protection Act (18.7%). The source PDFs have 402 and 40 usable pages respectively; page counts alone are not a quality or target quota. Do not assume all selected records are usable.

### Question types in the 1,000 selected candidates

| Type | Count | Share |
|---|---:|---:|
| rights_duties | 568 | 56.8% |
| condition_exception | 163 | 16.3% |
| direct_factual | 118 | 11.8% |
| scenario | 71 | 7.1% |
| definition | 52 | 5.2% |
| procedure_remedy | 27 | 2.7% |
| comparison | 1 | 0.1% |

`rights_duties` dominates. `procedure_remedy` and `definition` are relatively sparse; `comparison` has only one candidate. Do not force a question type where the source does not support it. Consumer Act coverage is especially sparse for procedure/remedy (7) and definitions (10); the Constitution has only one comparison candidate.

### Source/type coverage

- **constitution_of_india.pdf:** condition_exception 122, rights_duties 456, direct_factual 118, scenario 54, definition 42, procedure_remedy 20, comparison 1
- **consumer_protection_act_2019.pdf:** condition_exception 41, rights_duties 112, scenario 17, definition 10, procedure_remedy 7

The current set has 445 candidates with explicit numbered headings and 555 unconfirmed references. "Explicit heading" is only a citation lead, not legal validation. The 1,000 selected rows also contain 251 repeated-exact-answer and 253 repeated-answer-template flags (flags overlap). The first Stage 22 review set has one duplicate-flagged record; its raw worksheet reveals a near-identical answer record omitted by the selection cap. Resolve groups across the full candidate pool, not just the selected subset.

## Recommended construction and review pipeline

1. **Generate (unreviewed):** Extract complete, page-addressable provision units, not arbitrary mid-provision fragments. Ask deterministic, source-appropriate templates only when a full passage supports them. Preserve immutable candidate ID, source filename, PDF page, provision identifier, exact evidence, and the PDF version/hash. Keep domain as an explicit field: `constitution_of_india` or `consumer_protection_act_2019`; never merge their citations or silently transfer a reference between sources.
2. **Provenance-checked:** Check that passage text occurs on the stated page, inspect page boundaries/layout against the PDF, and independently confirm the provision heading/subsection. A page-text match does not check completeness or law. Leave references `UNCONFIRMED` if the heading cannot be tied directly to the passage.
3. **AI-screened (advisory only):** Flag answer/question mismatch, missing conditions, extraction artifacts, duplicate candidates, and source-reference mismatch. Store recommendation, evidence, and uncertainty separately. Never translate a model score or screening label into a review decision.
4. **Duplicate-screened:** First normalize exact questions conservatively and compare exact answer/question pairs. Then create near-duplicate groups using question similarity, answer similarity, provision/source location, and intent. Keep distinct questions with shared statutory language; flag group members together and have a person select/repair a representative. Run the checks across raw, deduplicated, selected, and proposed final records so candidate-limit omissions do not hide duplicates.
5. **Human-reviewed:** A reviewer reads the full provision plus surrounding clauses and checks the question, complete answer, qualifications, exception, and citation against the original. Require reviewer identifier, decision, concise reason, and explicit answer-support/reference confirmations. `EDIT_AND_RECHECK` stays pending until the revised answer and reference are checked again.
6. **Approved for dataset:** Only a distinct record with a completed human approval, reason, source-page/provision confirmation, complete supported answer, stable source metadata, and passed schema/provenance/duplicate checks enters the candidate final pool. Exclude unresolved OCR, citation, scope, duplicate, current-law, or interpretation issues. If the dataset claims current law, check authoritative current sources in a separate evidence field/review step; this audit did not do that. Keep a versioned rejection/defer ledger rather than deleting candidates.
7. **Build to 500 with measured coverage:** Set a target matrix by source and useful task types after reviewing provision inventory and human-approved yield; do not allocate quotas based solely on page ratios. First improve Consumer Act definitions and procedures, and Constitution procedure/definition coverage where source supports it. Batch 25-50 at a time, report generation -> provenance -> screen -> human decision -> approved counts separately, and only generate new candidates for uncovered provisions/types. Do not fill the count with repeats or weak categories.
8. **Split only after 500 approvals:** Deduplicate and group by provision/topic before splitting; keep near-duplicate families and source passages in one partition to reduce leakage. A later 400/50/50 split must be reproducible, seeded, recorded, and performed only after the genuinely approved unique count reaches 500.

## Required status model

Use separate fields/states, not one overloaded `review_status`:

- `GENERATED`: template candidate exists; unverified.
- `PROVENANCE_CHECKED`: source/page/evidence are traceable; no correctness claim.
- `AI_SCREENED`: advisory flags/recommendation recorded, still not a human decision.
- `HUMAN_REVIEWED`: explicit human decision and reason exist; not necessarily approved.
- `APPROVED_FOR_DATASET`: all required human confirmations, answer revalidation, and deterministic checks passed; unresolved flags absent.

Track `legal_currentness` separately as `NOT_ASSESSED`, `CHECK_REQUIRED`, or a documented human-verified source state; do not infer it from the PDF text. Keep edit/recheck status explicit.

## Evidence needed for any record to count

- Original PDF filename/version and 1-based page; exact passage and provision heading/subsection, including continuation page if needed.
- A question that has one clear answer within that source; an answer preserving actors, thresholds, exceptions, time limits, cross-references, and scope that the question asks about.
- Human-confirmed citation matches the cited provision; a heading or page match alone is insufficient.
- Duplicate-family review confirms a distinct useful item or records the chosen representative.
- Human decision, reviewer, reason, answer-supported confirmation, citation-source confirmation, plus recheck confirmation for edits.
- Passing schema/provenance checks and no unresolved extraction/scope/citation flags. Current-law claims need a separately cited authoritative current source and reviewer check.

## Current status and immediate next step

Stage 21 reconciled 1,000 selected review candidates, not 1,000 usable or approved examples. Stage 22 groups 21 for prioritized review and 10 for further review; this Stage 23 audit flags issues within the ready group too. Stage 19 had 13 `PROPOSED_APPROVE` labels, but those are AI suggestions, not decisions. **The approved count remains zero.** Continue by human-reviewing the 21 one at a time from the source PDFs, resolve the 10 listed cases separately, and record outcomes only through the existing review workflow. No final dataset or split should be made yet.
