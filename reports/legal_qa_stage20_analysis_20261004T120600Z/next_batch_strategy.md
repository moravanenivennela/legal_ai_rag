# Stage 20: First-pass failure analysis and next-batch strategy

## First-pass recommendations

The first pass contains 13 `PROPOSED_APPROVE`, 26 `PROPOSED_REJECT`, and 11 `PENDING_HUMAN_REVIEW` recommendations. These are AI screening labels only. All 50 source forms still say `PENDING`; no reviewer, reason, source confirmation, or approval was entered.

- Proposed-approve IDs: 037ef144b3e5c1ea, f5647ecdfb8edd4f, fa63b0dfbf03cfe8, fb671b1434235def, ed50b49df25f6058, ccf81f752f160b71, eafd86e56094bb25, dfe75d1781e7d3d1, cc07c5756e7d6e7b, f535f69c539348fd, dadaed71c47af958, ba8229bee203c7d2, f1942c4c4acec9a8
- Proposed-reject IDs: 004dd69be0a72aeb, edb58504d9963e5a, 0dd8417bbfe6d7bc, 0a9552ef52cfaff9, 1c53f50edc89caed, 014cfafe86aae67e, 06f032849da5dbd4, fe25da3c7158a625, e6f5df6480d94525, ffada58d009e6dde, fe7d73898eca8201, 64257f501275a7c1, fcfc86a574176015, f60b551073179690, d9dfd76da2fe00b5, e7b9568884b0bbc8, eab8471a2d4849b5, d7ba0f6ff8b6349a, f7dd9b76b71d5183, 43a1040f3801e6b3, e68a624739eb7a46, f6adb89377a3f28e, eab655c4291e3806, 42ece46ffcd8dd91, d620512e35308a6c, 40ae4bf48b2fdef8
- Pending IDs: 81ddc6564ee9b1bd, f8d48d6abe1c9354, e9786ae6f8f15f02, fbde0aa31a8f63b3, b268a450d367bf03, f15449dba95b5065, e7dda45e24d43b89, fbac661dc9c90700, fbd632de04371122, da206ca0d0dcda4c, c7c26340924bbb43

A proposed approval is not permission to train on a record. Human review against the original PDF is still required.

## Main failure patterns

Counts overlap: a row can be in more than one group. See the row-level CSV for all candidate IDs and specific flags.

- **wrong citation: 12 candidates** (overlapping): 004dd69be0a72aeb, 0a9552ef52cfaff9, 1c53f50edc89caed, 014cfafe86aae67e, fe7d73898eca8201, fcfc86a574176015, f60b551073179690, e7b9568884b0bbc8, eab8471a2d4849b5, f7dd9b76b71d5183, f6adb89377a3f28e, 40ae4bf48b2fdef8
- **incomplete answer: 16 candidates** (overlapping): edb58504d9963e5a, 0dd8417bbfe6d7bc, 06f032849da5dbd4, fe25da3c7158a625, f8d48d6abe1c9354, e6f5df6480d94525, ffada58d009e6dde, 64257f501275a7c1, d9dfd76da2fe00b5, d7ba0f6ff8b6349a, fbac661dc9c90700, 43a1040f3801e6b3, e68a624739eb7a46, eab655c4291e3806, 42ece46ffcd8dd91, 40ae4bf48b2fdef8
- **poorly formed question: 12 candidates** (overlapping): 0a9552ef52cfaff9, f8d48d6abe1c9354, e6f5df6480d94525, b268a450d367bf03, f60b551073179690, d9dfd76da2fe00b5, eab8471a2d4849b5, d7ba0f6ff8b6349a, f7dd9b76b71d5183, 43a1040f3801e6b3, e68a624739eb7a46, d620512e35308a6c
- **missing reference: 13 candidates** (overlapping): 81ddc6564ee9b1bd, fe25da3c7158a625, fbde0aa31a8f63b3, 64257f501275a7c1, d9dfd76da2fe00b5, d7ba0f6ff8b6349a, e7dda45e24d43b89, fbac661dc9c90700, e68a624739eb7a46, fbd632de04371122, d620512e35308a6c, da206ca0d0dcda4c, c7c26340924bbb43
- **duplicate or repeated answer: 6 candidates** (overlapping): 81ddc6564ee9b1bd, e6f5df6480d94525, e9786ae6f8f15f02, ffada58d009e6dde, e7b9568884b0bbc8, f7dd9b76b71d5183
- **ocr or extraction issue: 3 candidates** (overlapping): 1c53f50edc89caed, 014cfafe86aae67e, c7c26340924bbb43
- **uncertain legal interpretation: 5 candidates** (overlapping): 81ddc6564ee9b1bd, b268a450d367bf03, f15449dba95b5065, e7dda45e24d43b89, da206ca0d0dcda4c

The recurring issues are wrong provision labels, answers cut off or copied as fragments, and generated questions that do not clearly ask for the answer they contain. Duplicate/template warnings are signals for grouped review, not automatic deletion. Unconfirmed-reference status is missing citation metadata, not a license to infer the provision.

## Suggested wording repairs

The corrections CSV has 31 possible wording/citation repairs. Every row is `SUGGESTED_ONLY_NOT_APPROVED` and keeps the original ID, text, source filename, page, and complete original supporting passage. These are not imported into forms and do not count as reviewed examples. A duplicate warning remains even if a question can be rewritten. Footnote-only material and current-law uncertainty are not cured by paraphrasing.

## Practical path toward 500

1. Complete this first 50 through the existing human workflow. Use the suggested text only as a draft to compare against the PDF; the reviewer must make an independent decision and supply the required reason/source confirmation. Edited answers remain pending until rechecked.
2. Reconcile counts before drawing another batch. Stage 13 `run_manifest.json` says 1,000 selected, but its top-level status count is 2,424 pending (raw count); compare to the Stage 14 selected review workspace. The selected set is also reported as 813 Constitution and 187 Consumer Act records, so report actual balance rather than force equal quotas.
3. Prepare subsequent batches of 50?75 remaining candidates. Prioritize complete text with a clear provision heading and page, then vary source and supported question type. Keep uncertain references, duplicates, footnotes, and OCR/layout problems visibly flagged for later review.
4. Improve candidate construction at the source: build questions from a complete provision/definition/condition, preserve actor and qualifying language, exclude contents pages and amendment-history notes unless explicitly labeled as such, and never let an excerpt end mid-sentence. Keep deterministic outputs pending.
5. Apply exact-question deduplication and conservative near-duplicate grouping before review. Flag shared answer templates without deleting different questions merely for common statutory language. Re-check duplicates after edits.
6. Count toward 500 only distinct records with completed human decision, checked original provision/page, final answer revalidation after edits, and deterministic schema/provenance validation. Maintain an auditable approval ledger.
7. Only after the count reaches 500, create a separately versioned 400/50/50 split, grouping by provision before splitting to reduce leakage. This stage did not create that split.

## Limitations

Only the first-pass report, original blank forms, and the two supplied PDFs were used. No external/current-law source was consulted. PDF text matching is not legal correctness. Article 222 requires authoritative current-law checking due to the source footnote. No candidate was human-reviewed, approved, or legally verified in this stage; the 500 target is not met.
