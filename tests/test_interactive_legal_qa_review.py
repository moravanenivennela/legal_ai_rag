"""Focused tests for the non-destructive individual-review CLI."""

import sys
import copy
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.interactive_legal_qa_review import (
    collect_review_decision,
    render_candidate,
)


class InteractiveLegalQaReviewTests(unittest.TestCase):
    def setUp(self):
        self.form = {
            "candidate": {
                "candidate_id": "candidate-001",
                "question": "What does the provision state?",
                "proposed_answer": "The State shall act according to law.",
                "supporting_passage": (
                    "12. Example heading. The State shall act according to law."
                ),
                "source_document": "constitution_of_india.pdf",
                "pdf_page": 12,
                "legal_reference": "Article 12",
                "reference_status": "EXPLICIT_NUMBERED_HEADING",
                "reference_confidence_flags": (
                    "EXPLICIT_HEADING_MATCH_NOT_LEGAL_APPROVAL"
                ),
                "evidence_status": "DECLARED_PAGE_MATCH",
                "question_type": "direct_factual",
                "duplicate_flags": "REPEATED_ANSWER_TEMPLATE",
                "duplicate_candidate_ids": "candidate-002",
                "answer_template_repetition": True,
            },
            "review": {
                "decision": "PENDING",
                "reviewer": "",
                "reason": "",
                "answer_supported_by_passage_confirmation": "NO",
                "legal_reference_checked_against_original_source": "NO",
                "edited_answer_rechecked_confirmation": "NO",
                "edited_answer": "",
                "reviewed_legal_reference": "",
                "notes": "",
            },
        }

    def mock_input(self, values):
        iterator = iter(values)
        return lambda _prompt="": next(iterator)

    def test_candidate_display_includes_evidence_provenance_and_check_steps(self):
        text = render_candidate(
            self.form,
            1,
            50,
            Path("data/constitution_of_india.pdf"),
        )
        for expected in (
            "candidate-001",
            "What does the provision state?",
            "The State shall act according to law.",
            "12. Example heading.",
            "constitution_of_india.pdf",
            "PDF page: 12",
            "Article 12",
            "REPEATED_ANSWER_TEMPLATE",
            "Ctrl+f",
        ):
            self.assertIn(expected, text)

    def test_pending_choice_keeps_review_fields_blank(self):
        result = collect_review_decision(
            self.form,
            self.mock_input(["PENDING"]),
            lambda _text: None,
        )
        self.assertEqual(result["review"]["decision"], "PENDING")
        self.assertEqual(result["review"]["reviewer"], "")
        self.assertEqual(result["review"]["reason"], "")
        self.assertEqual(
            result["review"]["answer_supported_by_passage_confirmation"],
            "NO",
        )

    def test_approval_requires_both_confirmations_and_reason(self):
        original = copy.deepcopy(self.form)
        result = collect_review_decision(
            self.form,
            self.mock_input([
                "APPROVE",
                "Reviewer A",
                "Checked against the passage and source.",
                "YES",
                "YES",
            ]),
            lambda _text: None,
        )
        self.assertEqual(result["review"]["decision"], "APPROVE")
        self.assertEqual(result["review"]["answer_supported_by_passage_confirmation"], "YES")
        self.assertEqual(result["review"]["legal_reference_checked_against_original_source"], "YES")
        self.assertEqual(result["review"]["reason"], "Checked against the passage and source.")
        self.assertEqual(self.form, original)

    def test_missing_source_confirmation_returns_to_explicit_pending_choice(self):
        messages = []
        result = collect_review_decision(
            self.form,
            self.mock_input([
                "APPROVE",
                "Reviewer A",
                "Could not confirm citation.",
                "YES",
                "NO",
                "PENDING",
            ]),
            messages.append,
        )
        self.assertEqual(result["review"]["decision"], "PENDING")
        self.assertTrue(any("cannot be saved as entered" in line for line in messages))

    def test_unconfirmed_reference_requires_reviewer_entered_source_reference(self):
        form = copy.deepcopy(self.form)
        form["candidate"]["legal_reference"] = ""
        form["candidate"]["reference_status"] = "UNCONFIRMED"
        result = collect_review_decision(
            form,
            self.mock_input([
                "APPROVE",
                "Reviewer A",
                "Checked source.",
                "Article 12",
                "YES",
                "YES",
            ]),
            lambda _text: None,
        )
        self.assertEqual(
            result["review"]["reviewed_legal_reference"],
            "Article 12",
        )

    def test_edit_requires_reason_and_multiline_answer_but_is_not_approved(self):
        result = collect_review_decision(
            self.form,
            self.mock_input([
                "EDIT_AND_RECHECK",
                "Reviewer A",
                "Clarify wording.",
                "Revised answer text",
                "with a second line",
                ".",
            ]),
            lambda _text: None,
        )
        self.assertEqual(result["review"]["decision"], "EDIT_AND_RECHECK")
        self.assertEqual(
            result["review"]["edited_answer"],
            "Revised answer text\nwith a second line",
        )
        self.assertEqual(
            result["review"]["answer_supported_by_passage_confirmation"],
            "NO",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
