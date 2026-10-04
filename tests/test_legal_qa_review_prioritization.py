"""Tests for deterministic legal QA review prioritization."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_source_grounded_qa_candidates import HUMAN_REVIEW_STATUS, normalize_text
from scripts.prioritize_legal_qa_review import (
    build_report,
    select_first_review_batch,
)


def candidate(
    candidate_id,
    source,
    question_type,
    score=100,
    group="",
    related="",
):
    return {
        "candidate_id": candidate_id,
        "source_filename": source,
        "question_type": question_type,
        "priority_score": score,
        "duplicate_group_id": group,
        "duplicate_candidate_ids": related,
        "duplicate_flags": [],
        "review_status": HUMAN_REVIEW_STATUS,
        "review_decision": "",
        "reviewer": "",
        "reason": "",
        "reviewer_notes": "",
    }


class LegalQaReviewPrioritizationTests(unittest.TestCase):
    def test_batch_selection_is_deterministic(self):
        rows = [
            candidate(
                f"id-{index}",
                "constitution_of_india.pdf" if index < 6
                else "consumer_protection_act_2019.pdf",
                ("rights_duties", "definition", "scenario")[index % 3],
                score=100 - index,
            )
            for index in range(10)
        ]
        first = select_first_review_batch(rows, 5)
        second = select_first_review_batch(rows, 5)
        self.assertEqual(
            [row["candidate_id"] for row in first],
            [row["candidate_id"] for row in second],
        )

    def test_batch_seeds_both_sources_and_question_types(self):
        rows = [
            candidate(
                "c1",
                "constitution_of_india.pdf",
                "rights_duties",
                score=100,
            ),
            candidate(
                "c2",
                "constitution_of_india.pdf",
                "definition",
                score=95,
            ),
            candidate(
                "c3",
                "consumer_protection_act_2019.pdf",
                "procedure_remedy",
                score=90,
            ),
            candidate(
                "c4",
                "consumer_protection_act_2019.pdf",
                "scenario",
                score=85,
            ),
        ]
        selected = select_first_review_batch(rows, 4)
        self.assertEqual(
            {row["source_filename"] for row in selected},
            {
                "constitution_of_india.pdf",
                "consumer_protection_act_2019.pdf",
            },
        )
        self.assertEqual(
            {row["question_type"] for row in selected},
            {"rights_duties", "definition", "procedure_remedy", "scenario"},
        )

    def test_larger_batch_balances_source_and_question_type_coverage(self):
        rows = [
            candidate(
                f"const-{index:03}",
                "constitution_of_india.pdf",
                ("rights_duties", "definition", "scenario")[index % 3],
                score=100 - index % 10,
            )
            for index in range(40)
        ] + [
            candidate(
                f"consumer-{index:03}",
                "consumer_protection_act_2019.pdf",
                ("rights_duties", "definition", "scenario")[index % 3],
                score=100 - index % 10,
            )
            for index in range(10)
        ]
        selected = select_first_review_batch(rows, 20)
        source_counts = {}
        type_counts = {}
        for row in selected:
            source_counts[row["source_filename"]] = (
                source_counts.get(row["source_filename"], 0) + 1
            )
            type_counts[row["question_type"]] = (
                type_counts.get(row["question_type"], 0) + 1
            )
        self.assertEqual(source_counts["constitution_of_india.pdf"], 16)
        self.assertEqual(source_counts["consumer_protection_act_2019.pdf"], 4)
        self.assertEqual(set(type_counts), {"rights_duties", "definition", "scenario"})
        self.assertLessEqual(max(type_counts.values()) - min(type_counts.values()), 1)

    def test_duplicate_group_selects_one_best_representative(self):
        rows = [
            candidate("dup-a", "constitution_of_india.pdf", "rights_duties", 80, "g1", "dup-b"),
            candidate("dup-b", "constitution_of_india.pdf", "rights_duties", 95, "g1", "dup-a"),
            candidate("other", "consumer_protection_act_2019.pdf", "definition", 70),
        ]
        selected = select_first_review_batch(rows, 3)
        selected_ids = {row["candidate_id"] for row in selected}
        self.assertIn("dup-b", selected_ids)
        self.assertNotIn("dup-a", selected_ids)
        self.assertIn("other", selected_ids)

    def test_batch_records_remain_pending_without_decisions(self):
        rows = [
            candidate("c1", "constitution_of_india.pdf", "rights_duties"),
            candidate("c2", "consumer_protection_act_2019.pdf", "definition"),
        ]
        selected = select_first_review_batch(rows, 2)
        self.assertTrue(all(row["review_status"] == HUMAN_REVIEW_STATUS for row in selected))
        self.assertTrue(all(row["review_decision"] == "" for row in selected))
        self.assertTrue(all(row["reviewer"] == "" for row in selected))

    def test_report_flags_source_page_and_reference_statuses(self):
        evidence = (
            "14. Equality before law. The State shall not deny equality "
            "before the law within India and shall provide equal protection "
            "of the laws to every person."
        )
        records = [
            {
                "candidate_id": "verified",
                "passage_id": "p1",
                "question": "What does Article 14 provide?",
                "answer": "The State shall not deny equality before the law.",
                "evidence": evidence,
                "source_filename": "constitution_of_india.pdf",
                "page": 1,
                "legal_reference": "Article 14",
                "reference_status": "EXPLICIT_NUMBERED_HEADING",
                "question_type": "direct_factual",
                "review_status": HUMAN_REVIEW_STATUS,
                "duplicate_flags": [],
            },
            {
                "candidate_id": "unconfirmed",
                "passage_id": "p2",
                "question": "Which requirement is stated?",
                "answer": "An answer supported here.",
                "evidence": "Some source passage without a typed provision heading.",
                "source_filename": "consumer_protection_act_2019.pdf",
                "page": 1,
                "legal_reference": None,
                "reference_status": "UNCONFIRMED",
                "question_type": "rights_duties",
                "review_status": HUMAN_REVIEW_STATUS,
                "duplicate_flags": [],
            },
        ]
        worksheet = [
            {
                "candidate_id": row["candidate_id"],
                "duplicate_flags": "",
                "duplicate_group_id": "",
                "duplicate_candidate_ids": "",
                "reference_confidence_flags": (
                    "EXPLICIT_HEADING_MATCH_NOT_LEGAL_APPROVAL"
                    if row["legal_reference"]
                    else "REFERENCE_UNCONFIRMED"
                ),
                "review_decision": "",
                "reviewer": "",
                "reason": "",
                "reviewer_notes": "",
            }
            for row in records
        ]
        annotated, batch, summary = build_report(
            records,
            worksheet,
            {
                "constitution_of_india.pdf": [normalize_text(evidence)],
                "consumer_protection_act_2019.pdf": [
                    normalize_text(
                        "Some source passage without a typed provision heading."
                    )
                ],
            },
            batch_size=2,
        )
        by_id = {row["candidate_id"]: row for row in annotated}
        self.assertEqual(by_id["verified"]["evidence_status"], "DECLARED_PAGE_MATCH")
        self.assertEqual(
            by_id["verified"]["reference_evidence_status"],
            "EXPLICIT_REFERENCE_CONSISTENT",
        )
        self.assertEqual(
            by_id["unconfirmed"]["reference_evidence_status"],
            "UNCONFIRMED",
        )
        self.assertEqual(len(batch), 2)
        self.assertEqual(summary["review_decision_count"], 0)
        self.assertEqual(
            {row["review_status"] for row in batch},
            {HUMAN_REVIEW_STATUS},
        )
        self.assertTrue(all(row["review_decision"] == "" for row in batch))


if __name__ == "__main__":
    unittest.main(verbosity=2)
