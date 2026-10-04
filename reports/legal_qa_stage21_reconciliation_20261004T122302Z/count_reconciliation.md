# Stage 21: Candidate count reconciliation

## Reconciled counts

| Stage/meaning | Count | What the number represents |
|---|---:|---|
| Raw candidates | 2,424 | Deterministic candidate rows in the full raw review worksheet, before question deduplication. |
| Distinct normalized questions | 2,119 | Distinct after `normalize_text` (Unicode NFKC, case-fold, alphanumeric tokens); not literal byte-for-byte strings. |
| Deduplicated pool | 2,113 | Valid candidates remaining after duplicate removal, before the selection cap. |
| Selected candidates | 1,000 | Rows in `candidates.jsonl`, selected under the 1,000-candidate limit. |
| Stage 14 review workspace | 1,000 | Selected records loaded into the separate review workspace; all pending. |
| Stage 15 all-candidate prioritization | 1,000 | Review priority metadata for the selected records. |
| Stage 15 first batch | 50 | Queue for human review, not reviewed or approved records. |
| Stage 16 individual forms | 50 | Blank candidate forms; states: `{'PENDING': 50}`. |
| Stage 17 saved forms | 1 in first session; 0 in second | First session saved one `PENDING` form; second output directory contains no decision forms. No approval was recorded. |
| Human-approved examples | 0 | No approved example was found in these artifacts. Stage 19 proposed labels are AI screening recommendations, not decisions. |

## Why the counts differ

The Stage 13 script counts **2,424 raw rows**. It normalizes question strings before computing **2,119 distinct questions**. Its deduplication removed **311 rows** (304 exact-question duplicate flags and 7 near-question duplicate flags), leaving **2,113**. The arithmetic difference between raw rows and unique normalized questions is 305; the remaining six removals are distinct normalized strings removed by the near-duplicate similarity rule. The script retains the first matching/near-matching question according to its deterministic ordering, while answer-template repetition is flagged separately.

The candidate limit then selected **1,000** of the 2,113 deduplicated candidates and omitted **1,113**. The Stage 14 workspace and Stage 15 prioritization both use those 1,000; Stage 15 selected 50 for the first batch (41 Constitution, 9 Consumer Act). These are review-eligible/queued records, not vetted records.

## Consistency finding

There is **no arithmetic inconsistency** in 2,424 -> 2,119 -> 2,113 -> 1,000 -> 1,000 -> 50. The genuine reporting-scope hazard is in the Stage 13 manifest: `counts.retained_candidates` and `candidates.jsonl` refer to the 1,000 selected rows, while `counts.human_review_pending`, top-level `review_status_counts`, and `human_review_report.csv` refer to all 2,424 raw rows. Both collections are pending, but the different denominators should be labeled explicitly. Also, `target_500_reached: true` is calculated from selected candidate count (at least 500), not 500 human-approved examples.

## State of human review

Stage 14 reports 1,000 pending and zero approved. Stage 15 has 50 pending review-queue rows. All 50 Stage 16 forms are pending. The inspected Stage 17 outputs contain one saved `PENDING` form and an empty second decision directory. Stage 19 has 13 `PROPOSED_APPROVE`, 26 `PROPOSED_REJECT`, and 11 `PENDING_HUMAN_REVIEW` AI recommendations; these do not change human-review status. Stage 20 explicitly records zero human decisions and approvals.

**Conclusion:** 1,000 selected for review does not mean 1,000 reviewed; the first batch of 50 does not mean 50 reviewed; and the number of approved examples remains zero. No final dataset or 400/50/50 split exists from this work.
