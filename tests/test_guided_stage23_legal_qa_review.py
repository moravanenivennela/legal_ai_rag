"""Tests for the Stage 23 guided review session."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.guided_stage23_legal_qa_review import (
    collect_decision,
    load_review_candidates,
    render_candidate,
    validate_guided_decision,
    validate_source_mapping,
)


class GuidedStage23LegalQaReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "candidate_id": "candidate-001",
            "original_question": "What does Article 12 say?",
            "original_answer": "The State shall act.",
            "proposed_question": "What does Article 12 provide?",
            "proposed_answer": "The State shall act.",
            "original_reference": "Article 11",
            "proposed_reference": "Article 12",
            "source_document": "constitution_of_india.pdf",
            "pdf_page_1_based": 12,
            "source_provision_excerpt": "Article 12. The State shall act.",
            "adjacent_page_excerpt_if_needed": "",
            "source_text_audit": "SOURCE_TEXT_ALIGNED",
            "source_text_finding": "Source text excerpt is present.",
            "reference_page_consistency": "PROVISION_TEXT_ON_CITED_PAGE",
            "quality_or_scope_flags": "Check surrounding qualification.",
            "duplicate_flags": "REPEATED_ANSWER_TEMPLATE",
            "ai_recommendation": "PROPOSED_APPROVE",
        }

    @staticmethod
    def mock_input(values: list[str]):
        iterator = iter(values)
        return lambda *_args: next(iterator)

    def valid_approval(self) -> dict:
        return {
            "candidate": deepcopy(self.record),
            "review": {
                "decision": "APPROVE",
                "reviewer": "Reviewer A",
                "reason": "Checked full provision and source reference.",
                "source_provision_inspected_confirmation": "YES",
                "legal_reference_inspected_confirmation": "YES",
                "answer_supported_by_complete_provision_confirmation": "YES",
                "reviewed_reference": "Article 12",
                "reviewed_text_source": "SUGGESTED",
                "reviewed_question": "What does Article 12 provide?",
                "reviewed_answer": "The State shall act.",
                "edited_answer_rechecked_confirmation": "NO",
                "dataset_eligibility": (
                    "HUMAN_APPROVAL_RECORDED_NOT_LEGAL_VERIFICATION"
                ),
            },
        }

    def test_approval_requires_reviewer_reason_and_all_source_confirmations(self):
        record = self.valid_approval()
        record["review"]["reviewer"] = ""
        record["review"]["reason"] = ""
        record["review"]["source_provision_inspected_confirmation"] = "NO"
        errors = validate_guided_decision(record, "candidate-001")
        self.assertTrue(any("Reviewer is required" in error for error in errors))
        self.assertTrue(any("reason is required" in error for error in errors))
        self.assertTrue(any("source provision" in error for error in errors))

    def test_approval_requires_explicit_reference_and_complete_provision_support(self):
        record = self.valid_approval()
        record["review"]["reviewed_reference"] = ""
        record["review"]["answer_supported_by_complete_provision_confirmation"] = "NO"
        errors = validate_guided_decision(record, "candidate-001")
        self.assertTrue(any("exact reference" in error for error in errors))
        self.assertTrue(any("complete provision" in error for error in errors))

    def test_reviewer_edit_cannot_be_approved_without_independent_recheck(self):
        record = self.valid_approval()
        record["review"]["reviewed_text_source"] = "REVIEWER_EDIT"
        record["review"]["edited_answer_rechecked_confirmation"] = "NO"
        self.assertTrue(any(
            "independent source recheck" in error
            for error in validate_guided_decision(record, "candidate-001")
        ))

    def test_pending_record_is_not_eligible(self):
        record = {
            "candidate": deepcopy(self.record),
            "review": {
                "decision": "PENDING",
                "answer_supported_by_complete_provision_confirmation": "NO",
                "legal_reference_inspected_confirmation": "NO",
                "edited_answer_rechecked_confirmation": "NO",
                "dataset_eligibility": "INELIGIBLE",
            },
        }
        self.assertEqual(validate_guided_decision(record, "candidate-001"), [])
        result = collect_decision(
            self.record,
            self.mock_input(["PENDING"]),
            lambda *_args: None,
        )
        self.assertEqual(result["review"]["decision"], "PENDING")
        self.assertEqual(result["review"]["dataset_eligibility"], "INELIGIBLE")

    def test_edit_and_recheck_is_pending_independent_recheck_and_approval(self):
        record = {
            "candidate": deepcopy(self.record),
            "review": {
                "decision": "EDIT_AND_RECHECK",
                "reviewer": "Reviewer A",
                "reason": "Improve clarity.",
                "edited_answer": "Revised answer.",
                "answer_supported_by_complete_provision_confirmation": "NO",
                "legal_reference_inspected_confirmation": "NO",
                "edited_answer_rechecked_confirmation": "NO",
                "dataset_eligibility": (
                    "INELIGIBLE_PENDING_INDEPENDENT_RECHECK_AND_APPROVAL"
                ),
            },
        }
        self.assertEqual(validate_guided_decision(record, "candidate-001"), [])
        record["review"]["dataset_eligibility"] = (
            "HUMAN_APPROVAL_RECORDED_NOT_LEGAL_VERIFICATION"
        )
        self.assertTrue(validate_guided_decision(record, "candidate-001"))

    def test_valid_approval_is_only_a_review_decision_not_a_legal_claim(self):
        record = self.valid_approval()
        self.assertEqual(validate_guided_decision(record, "candidate-001"), [])
        result = collect_decision(
            self.record,
            self.mock_input([
                "APPROVE",
                "Reviewer A",
                "Checked the entire provision.",
                "YES",
                "YES",
                "YES",
                "Article 12",
                "SUGGESTED",
                "What does Article 12 provide?",
                ".",
                "The State shall act.",
                ".",
            ]),
            lambda *_args: None,
        )
        self.assertEqual(result["review"]["decision"], "APPROVE")
        self.assertIn("not a guarantee", result["legal_correctness_notice"])
        self.assertEqual(
            result["review"]["dataset_eligibility"],
            "HUMAN_APPROVAL_RECORDED_NOT_LEGAL_VERIFICATION",
        )

    def test_incomplete_approval_is_not_silently_downgraded_to_pending(self):
        messages: list[str] = []
        result = collect_decision(
            self.record,
            self.mock_input([
                "APPROVE",
                "",
                "",
                "NO",
                "NO",
                "NO",
                "",
                "ORIGINAL",
                "",
                ".",
                "",
                ".",
                "PENDING",
            ]),
            messages.append,
        )
        self.assertEqual(result["review"]["decision"], "PENDING")
        self.assertTrue(any("No decision was saved" in line for line in messages))

    def test_collecting_decision_does_not_mutate_original_candidate(self):
        original = deepcopy(self.record)
        collect_decision(
            self.record,
            self.mock_input(["PENDING"]),
            lambda *_args: None,
        )
        self.assertEqual(self.record, original)

    def test_review_card_shows_original_suggestion_flags_ai_label_and_pdf_context(self):
        text = render_candidate(
            self.record,
            1,
            21,
            Path("constitution_of_india.pdf"),
            [(11, "Previous qualification."), (12, "Article 12. Full source text.")],
        )
        for expected in (
            "What does Article 12 say?",
            "What does Article 12 provide?",
            "Original reference: Article 11",
            "Proposed reference: Article 12",
            "PROPOSED_APPROVE",
            "not a legal conclusion",
            "Check surrounding qualification.",
            "REPEATED_ANSWER_TEMPLATE",
            "FULL PDF PAGE CONTEXT",
            "PDF PAGE 11",
            "Previous qualification.",
        ):
            self.assertIn(expected, text)

    def test_source_mapping_rejects_wrong_document_or_out_of_range_page(self):
        page_counts = {"constitution_of_india.pdf": 402}
        validate_source_mapping("constitution_of_india.pdf", 402, page_counts)
        with self.assertRaises(ValueError):
            validate_source_mapping("../private.pdf", 1, page_counts)
        with self.assertRaises(ValueError):
            validate_source_mapping("constitution_of_india.pdf", 403, page_counts)

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        fields = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _inputs(self, root: Path, duplicate_audit_id: bool = False):
        audit_rows = []
        stage22_rows = []
        forms_dir = root / "forms"
        forms_dir.mkdir()
        for index in range(21):
            candidate_id = f"candidate-{index:03d}"
            if duplicate_audit_id and index == 20:
                candidate_id = "candidate-000"
            original_question = f"Question {index}?"
            original_answer = f"Answer {index}."
            audit_rows.append({
                "candidate_id": candidate_id,
                "source_document": "constitution_of_india.pdf",
                "pdf_page_1_based": "12",
                "original_question": original_question,
                "original_answer": original_answer,
                "proposed_question": original_question,
                "proposed_answer": original_answer,
                "original_reference": "Article 12",
                "proposed_reference": "Article 12",
                "source_text_audit": "SOURCE_TEXT_ALIGNED",
                "source_text_finding": "Source excerpt.",
                "reference_page_consistency": "PROVISION_TEXT_ON_CITED_PAGE",
                "quality_or_scope_flags": "",
                "duplicate_flags": "",
                "first_pass_recommendation": "PROPOSED_APPROVE",
                "source_provision_excerpt": "Article 12. Text.",
                "adjacent_page_excerpt_if_needed": "",
            })
            stage22_rows.append({
                "priority_group": "READY_FOR_HUMAN_REVIEW",
                "candidate_id": candidate_id,
                "original_question": original_question,
                "original_answer": original_answer,
                "proposed_question": original_question,
                "proposed_answer": original_answer,
                "proposed_reference": "Article 12",
                "source_document": "constitution_of_india.pdf",
                "pdf_page_1_based": "12",
            })
            form = {
                "candidate": {
                    "candidate_id": candidate_id,
                    "question": original_question,
                    "proposed_answer": original_answer,
                    "source_document": "constitution_of_india.pdf",
                    "pdf_page": "12",
                },
                "review": {"decision": "PENDING"},
            }
            (forms_dir / f"{candidate_id}.json").write_text(
                json.dumps(form), encoding="utf-8"
            )
        self._write_csv(root / "audit.csv", audit_rows)
        self._write_csv(root / "stage22.csv", stage22_rows)
        return root / "audit.csv", root / "stage22.csv", forms_dir

    def test_loader_validates_source_mapping_and_preserves_original_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, stage22, forms = self._inputs(root)
            originals = {
                path.name: path.read_bytes()
                for path in [*forms.glob("*.json"), audit, stage22]
            }
            records = load_review_candidates(
                audit, stage22, forms, {"constitution_of_india.pdf": 402}
            )
            self.assertEqual(len(records), 21)
            self.assertEqual(
                {path.name: path.read_bytes() for path in [*forms.glob("*.json"), audit, stage22]},
                originals,
            )

    def test_loader_rejects_duplicate_candidate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, stage22, forms = self._inputs(root, duplicate_audit_id=True)
            with self.assertRaisesRegex(ValueError, "duplicate candidate IDs"):
                load_review_candidates(
                    audit, stage22, forms, {"constitution_of_india.pdf": 402}
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
