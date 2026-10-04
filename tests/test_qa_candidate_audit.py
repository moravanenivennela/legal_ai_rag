"""Tests for read-only pending legal QA candidate audit helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_pending_qa_candidates import (
    audit_candidate,
    compare_duplicates,
    page_evidence_status,
    reference_status,
)


class QaCandidateAuditTests(unittest.TestCase):
    def setUp(self):
        self.valid_record = {
            "_record_id": "train_candidate-0001",
            "instruction": "Answer from legal evidence.",
            "input": (
                "Question: What does Article 14 say?\n\n"
                "Legal evidence:\n14. Equality before law."
            ),
            "output": "Article 14 says equality before law.",
            "evidence": "14. Equality before law.",
            "source": "constitution",
            "page": "1",
            "legal_reference": "Article 14",
            "split": "train",
        }
        self.source_pages = {
            "constitution": [
                "14 equality before law",
            ],
            "consumer_protection": [
                "section 1 short title",
            ],
        }

    def test_page_evidence_match_and_mismatch_are_distinct(self):
        self.assertEqual(
            page_evidence_status(
                "14. Equality before law.",
                "constitution",
                "1",
                self.source_pages,
            )[0],
            "MATCH",
        )
        self.assertEqual(
            page_evidence_status(
                "Different evidence",
                "constitution",
                "1",
                self.source_pages,
            )[0],
            "NO_MATCH",
        )

    def test_reference_requires_literal_support_and_never_autocorrects(self):
        self.assertEqual(
            reference_status("Article 14", self.valid_record["evidence"])[0],
            "TYPE_OR_CONTEXT_UNCONFIRMED",
        )
        self.assertEqual(
            reference_status("Article 15", self.valid_record["evidence"])[0],
            "NOT_SUPPORTED_BY_EVIDENCE",
        )
        self.assertEqual(
            reference_status("PASSAGE", "This passage contains the word passage.")[0],
            "MALFORMED_OR_AMBIGUOUS",
        )
        self.assertEqual(
            reference_status("", self.valid_record["evidence"])[0],
            "MISSING",
        )

    def test_page_mismatch_is_needs_review_not_invalid(self):
        record = dict(self.valid_record)
        report = audit_candidate(record, {
            "constitution": ["different source page"],
            "consumer_protection": ["section 1 short title"],
        }, {}, {}, {})
        self.assertEqual(report["page_evidence_status"], "NO_MATCH")
        self.assertEqual(report["audit_status"], "NEEDS_REVIEW")

    def test_exact_and_near_duplicates_include_existing_records(self):
        existing = [{
            "question": "What does Article 14 say?",
            "answer": "Equality before law.",
        }]
        exact, answers, near = compare_duplicates(
            [self.valid_record], existing
        )
        self.assertIn("EXISTING-0001", exact["train_candidate-0001"])
        self.assertNotIn("EXISTING-0001", answers.get("train_candidate-0001", set()))
        self.assertEqual(near, {})

    def test_near_duplicate_question_is_flagged_against_existing(self):
        candidate = dict(self.valid_record)
        candidate["input"] = (
            "Question: What does Article 14 provide about equality before the law?\n\n"
            "Legal evidence:\n14. Equality before law."
        )
        existing = [{
            "question": "What does Article 14 say about equality before the law?",
            "answer": "A different answer.",
        }]
        _, _, near = compare_duplicates([candidate], existing)
        self.assertIn("EXISTING-0001", near["train_candidate-0001"])

    def test_malformed_candidate_is_invalid_record(self):
        malformed = {
            "_record_id": "train_candidate-0002",
            "_parse_error": "Malformed JSON",
        }
        report = audit_candidate(malformed, self.source_pages, {}, {}, {})
        self.assertEqual(report["audit_status"], "INVALID_RECORD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
