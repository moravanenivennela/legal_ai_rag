"""Deterministic tests for evidence normalization and page diagnostics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_qa_evidence import (
    inspect_reference,
    match_evidence_pages,
    normalize_text,
)


class EvidenceDiagnosticTests(unittest.TestCase):
    def test_normalization_handles_ligatures_hyphens_punctuation_and_lines(self):
        self.assertEqual(
            normalize_text("The beneﬁt is equal-\nity before law."),
            "the benefit is equality before law",
        )

    def test_matching_accepts_either_dehyphenated_or_separated_extraction(self):
        result = match_evidence_pages(
            "The inter-\nnational commission reviews the record.",
            1,
            ["The international commission reviews the record."],
        )
        self.assertEqual(result["match_status"], "DECLARED_PAGE_MATCH")

    def test_declared_page_match_is_reported_without_nearby_page_substitution(self):
        result = match_evidence_pages(
            "The State shall not deny equality before law.",
            2,
            [
                "Unrelated first page",
                "The State shall not deny equality before law.",
                "Repeated surrounding text",
            ],
        )
        self.assertEqual(result["match_status"], "DECLARED_PAGE_MATCH")
        self.assertTrue(result["declared_page_match"])
        self.assertEqual(result["matching_page_numbers"], [2])

    def test_nearby_page_match_does_not_change_declared_page(self):
        result = match_evidence_pages(
            "The State shall not deny equality before law.",
            2,
            [
                "Unrelated first page",
                "Unrelated declared page",
                "The State shall not deny equality before law.",
            ],
        )
        self.assertEqual(result["match_status"], "NEARBY_PAGE_MATCH")
        self.assertFalse(result["declared_page_match"])
        self.assertEqual(result["nearby_page_matches"], [3])

    def test_multiple_page_matches_are_ambiguous(self):
        passage = "Section 1 provides a short title"
        result = match_evidence_pages(
            passage,
            2,
            [
                passage,
                passage,
                "Another page",
                passage,
            ],
            nearby_radius=2,
        )
        self.assertEqual(result["match_status"], "AMBIGUOUS_MATCH")
        self.assertEqual(result["matching_page_numbers"], [1, 2, 4])

    def test_page_boundary_match_is_diagnostic_nearby_match(self):
        result = match_evidence_pages(
            "The State shall not deny equality before law",
            1,
            ["The State shall not deny", "equality before law"],
        )
        self.assertEqual(result["match_status"], "NEARBY_PAGE_MATCH")
        self.assertEqual(result["spanning_page_matches"], [[1, 2]])

    def test_reference_conflict_is_not_corrected(self):
        status, _ = inspect_reference(
            "Article 15",
            "14. Equality before law. The State shall not deny equality.",
        )
        self.assertEqual(status, "REFERENCE_CONFLICT_OR_UNCONFIRMED")

    def test_explicit_reference_match_ignores_case_and_punctuation(self):
        status, _ = inspect_reference(
            "Article 14",
            "14. Equality before law. See ARTICLE 14.",
        )
        self.assertEqual(status, "REFERENCE_MATCH")

    def test_prose_in_reference_field_is_not_treated_as_a_citation(self):
        passage = (
            "Article 14. The State shall not deny equality before the law."
        )
        status, reason = inspect_reference(passage, passage)
        self.assertEqual(status, "AMBIGUOUS_REFERENCE")
        self.assertIn("contains prose", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
