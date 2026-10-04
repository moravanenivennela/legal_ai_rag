"""Focused tests for conservative supported-candidate evidence triage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.legal_qa_audit_core import _heading_index
from scripts.verify_supported_candidate_evidence import review_candidate_evidence


class SupportedCandidateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.page_text = (
            "14. Equality before law.\n"
            "The State shall not deny to any person equality before the law."
        )
        self.record = {
            "candidate_id": "candidate-14",
            "automated_status": "SUPPORTED_CANDIDATE",
            "original_question": "What does Article 14 state?",
            "original_answer": (
                "The State shall not deny to any person equality before the law."
            ),
            "original_reference": "Article 14",
            "source_document": "constitution_of_india.pdf",
            "declared_pdf_page": 1,
            "exact_source_evidence_excerpt": self.page_text,
            "reason": "Audit text-matching rationale.",
            "issues": [],
            "confidence": 0.9,
            "confidence_scope": "deterministic text classification only",
            "suggested_question": "Suggested wording",
            "suggested_answer": "",
            "suggested_reference": "",
        }
        self.pages = {
            "constitution_of_india.pdf": [self.page_text],
            "consumer_protection_act_2019.pdf": [],
        }

    def review(self, record=None, pages=None):
        corpus = pages if pages is not None else self.pages
        return review_candidate_evidence(
            record or self.record,
            corpus,
            headings_by_source=_heading_index(corpus),
        )

    def test_full_answer_and_excerpt_are_reverified_on_declared_page(self):
        result = self.review()
        self.assertTrue(result["exact_source_text_match"])
        self.assertTrue(result["answer_verbatim_in_source_excerpt"])
        self.assertEqual(
            result["source_match_triage_status"],
            "EXACT_SOURCE_EVIDENCE_REQUIRES_REVIEW",
        )
        self.assertEqual(
            result["review_disposition"],
            "REQUIRES_INDEPENDENT_HUMAN_LEGAL_REVIEW",
        )
        self.assertEqual(
            result["answer_to_question_semantic_support"],
            "NOT_ESTABLISHED_AUTOMATICALLY",
        )
        self.assertEqual(result["legal_verification_status"], "NOT_LEGALLY_VERIFIED")

    def test_similar_but_nonverbatim_answer_requires_review(self):
        record = {
            **self.record,
            "original_answer": "The State must grant everyone equal protection.",
        }
        result = self.review(record)
        self.assertFalse(result["exact_source_text_match"])
        self.assertEqual(
            result["source_match_triage_status"],
            "SOURCE_MATCH_REQUIRES_REVIEW",
        )
        self.assertIn(
            "ANSWER_NOT_FOUND_VERBATIM_WITHIN_SOURCE_EXCERPT",
            result["review_flags"],
        )

    def test_ocr_artifact_and_unverified_reference_are_conservative(self):
        record = {
            **self.record,
            "original_question": "What does Article 14� state?",
            "original_reference": "Article 15",
        }
        result = self.review(record)
        self.assertEqual(
            result["source_match_triage_status"],
            "SOURCE_MATCH_REQUIRES_REVIEW",
        )
        self.assertIn(
            "POSSIBLE_OCR_OR_REPLACEMENT_CHARACTER_CORRUPTION",
            result["review_flags"],
        )
        self.assertEqual(
            result["reference_match_result"],
            "REFERENCE_UNRESOLVED",
        )

    def test_original_and_suggested_text_remain_separate(self):
        result = self.review()
        self.assertEqual(result["original_question"], self.record["original_question"])
        self.assertEqual(result["suggested_question"], "Suggested wording")
        self.assertEqual(result["original_answer"], self.record["original_answer"])

    def test_missing_pdf_page_uses_review_status(self):
        pages = {**self.pages, "constitution_of_india.pdf": None}
        result = self.review(pages=pages)
        self.assertEqual(
            result["source_match_triage_status"],
            "SOURCE_MATCH_REQUIRES_REVIEW",
        )
        self.assertIn("SOURCE_PDF_UNAVAILABLE", result["review_flags"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
