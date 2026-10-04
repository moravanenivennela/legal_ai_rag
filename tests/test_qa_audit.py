"""Tests for deterministic QA provenance-audit helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_existing_qa import (
    audit_record,
    duplicate_flags,
    locate_passage,
    normalize_text,
)


class QaAuditTests(unittest.TestCase):
    def test_normalized_passage_match_tolerates_pdf_whitespace_and_punctuation(self):
        self.assertTrue(
            locate_passage(
                "The State shall not deny to any person, equality before law.",
                "THE STATE shall not deny to any person,\nequality before law.",
            )
        )

    def test_page_mismatch_requires_review_not_rejection(self):
        record = {
            "question": "What does the passage state?",
            "answer": "The State shall act.",
            "legal_reference": "Not explicitly identified",
            "supporting_passage": "The State shall act.",
            "source": "constitution",
            "page": 1,
        }
        report = audit_record(
            record,
            0,
            {"constitution": ["a different page"], "consumer_protection": []},
            {("what does the passage state", "the state shall act"): "train"},
            {},
            {0: []},
            {0: []},
            {0: []},
        )
        self.assertEqual(report["audit_status"], "NEEDS_REVIEW")
        self.assertEqual(report["page_match"], "NO_NORMALIZED_MATCH")

    def test_exact_extract_with_traceable_page_and_no_reference_can_be_verified(self):
        record = {
            "question": "What does the passage state?",
            "answer": "The State shall act.",
            "legal_reference": "Not explicitly identified",
            "supporting_passage": "The State shall act.",
            "source": "constitution",
            "page": 1,
        }
        report = audit_record(
            record,
            0,
            {"constitution": ["Here, the State shall act."], "consumer_protection": []},
            {("what does the passage state", "the state shall act"): "train"},
            {},
            {0: []},
            {0: []},
            {0: []},
        )
        self.assertEqual(report["audit_status"], "VERIFIED")
        self.assertTrue(report["answer_is_extractive"])

    def test_unmatched_legal_reference_requires_review(self):
        report = audit_record(
            {
                "question": "What does Article 10 require?",
                "answer": "The State shall act.",
                "legal_reference": "Article 10",
                "supporting_passage": "The State shall act.",
                "source": "constitution",
                "page": 1,
            },
            0,
            {"constitution": ["The State shall act."], "consumer_protection": []},
            {},
            {},
            {0: []},
            {0: []},
            {0: []},
        )
        self.assertEqual(report["audit_status"], "NEEDS_REVIEW")
        self.assertEqual(report["reference_consistency"], "NOT_MATCHED")

    def test_duplicate_flags_are_stable_and_exact_question_normalization_works(self):
        records = [
            {"question": "What does Article 14 say?", "answer": "Equality before law."},
            {"question": "what does article 14 say", "answer": "Equality before law."},
            {"question": "What does Article 15 say?", "answer": "Different answer."},
        ]
        questions, answers, near = duplicate_flags(records)
        self.assertEqual(questions[0], [1])
        self.assertEqual(answers[0], [1])
        self.assertEqual(normalize_text(records[0]["question"]), normalize_text(records[1]["question"]))
        self.assertIn(1, near[0])
        self.assertIn(0, near[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
