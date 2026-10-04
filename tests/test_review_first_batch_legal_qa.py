"""Tests for the Stage 16 individual legal QA review workflow."""

import sys
import csv
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_first_batch_legal_qa import (
    _candidate_form_mismatches,
    process_review_forms,
    validate_approved_form,
    validate_decision_form,
)


class IndividualReviewWorkflowTests(unittest.TestCase):
    batch_fields = [
        "candidate_id",
        "question",
        "answer",
        "evidence",
        "source_filename",
        "page",
        "legal_reference",
        "reference_status",
        "reference_confidence_flags",
        "evidence_status",
        "question_type",
        "duplicate_group_id",
        "duplicate_candidate_ids",
        "duplicate_flags",
        "answer_template_repetition",
        "priority_rank",
        "priority_reasons",
        "review_status",
        "review_decision",
        "reviewer",
        "reason",
    ]

    def setUp(self):
        self.candidate = {
            "candidate_id": "candidate-1",
            "question": "What does Article 14 provide?",
            "answer": "The State shall not deny equality before the law.",
            "evidence": (
                "14. Equality before law. The State shall not deny equality "
                "before the law and shall provide equal protection of the laws."
            ),
            "source_filename": "constitution_of_india.pdf",
            "page": "37",
            "legal_reference": "Article 14",
            "proposed_answer": "The State shall not deny equality before the law.",
            "supporting_passage": (
                "14. Equality before law. The State shall not deny equality "
                "before the law and shall provide equal protection of the laws."
            ),
            "source_document": "constitution_of_india.pdf",
            "pdf_page": "37",
            "reference_status": "EXPLICIT_NUMBERED_HEADING",
            "reference_confidence_flags": (
                "EXPLICIT_HEADING_MATCH_NOT_LEGAL_APPROVAL"
            ),
            "evidence_status": "DECLARED_PAGE_MATCH",
            "question_type": "direct_factual",
            "duplicate_group_id": "",
            "duplicate_candidate_ids": "",
            "duplicate_flags": "",
            "answer_template_repetition": "False",
            "priority_rank": "1",
            "priority_reasons": "test evidence",
            "review_status": "PENDING_HUMAN_REVIEW",
            "review_decision": "",
            "reviewer": "",
            "reason": "",
        }
        self.form_candidate = {
            "candidate_id": self.candidate["candidate_id"],
            "question": self.candidate["question"],
            "proposed_answer": self.candidate["proposed_answer"],
            "supporting_passage": self.candidate["supporting_passage"],
            "source_document": self.candidate["source_document"],
            "pdf_page": self.candidate["pdf_page"],
            "legal_reference": self.candidate["legal_reference"],
            "reference_status": self.candidate["reference_status"],
            "reference_confidence_flags": self.candidate[
                "reference_confidence_flags"
            ],
            "evidence_status": self.candidate["evidence_status"],
            "question_type": self.candidate["question_type"],
            "duplicate_group_id": "",
            "duplicate_candidate_ids": "",
            "duplicate_flags": "",
            "answer_template_repetition": "False",
            "priority_rank": "1",
            "priority_reasons": "test evidence",
        }

    def form(self, **review_changes):
        review = {
            "decision": "PENDING",
            "reviewer": "",
            "reason": "",
            "answer_supported_by_passage_confirmation": "NO",
            "legal_reference_checked_against_original_source": "NO",
            "edited_answer_rechecked_confirmation": "NO",
            "edited_answer": "",
            "reviewed_legal_reference": "",
            "notes": "",
        }
        review.update(review_changes)
        return {"candidate": self.form_candidate, "review": review}

    def process(self, temp_path, review, directory_name):
        batch_path = temp_path / "batch.csv"
        with batch_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.batch_fields)
            writer.writeheader()
            writer.writerow({
                field: self.candidate.get(field, "")
                for field in self.batch_fields
            })
        forms_dir = temp_path / f"{directory_name}-forms"
        forms_dir.mkdir()
        form = self.form(**review)
        (forms_dir / "candidate-1.json").write_text(
            json.dumps(form),
            encoding="utf-8",
        )
        output_dir = temp_path / f"{directory_name}-output"
        source_pages = {
            (
                "constitution_of_india.pdf",
                37,
            ): self.candidate["supporting_passage"],
        }
        summary = process_review_forms(
            forms_dir,
            batch_path,
            output_dir,
            source_pages,
        )
        return summary, output_dir

    def test_incomplete_and_unknown_decisions_are_rejected(self):
        incomplete = self.form(decision="APPROVE")
        errors = validate_decision_form(incomplete, self.candidate)
        self.assertIn(
            "Reviewer is required for a completed decision.",
            errors,
        )
        self.assertIn(
            "A reason is required for a completed decision.",
            errors,
        )
        invalid = self.form(decision="MAYBE")
        self.assertIn(
            "Decision must be APPROVE, REJECT, EDIT_AND_RECHECK, or PENDING.",
            validate_decision_form(invalid, self.candidate),
        )

    def test_completed_decisions_require_reason(self):
        form = self.form(
            decision="REJECT",
            reviewer="Reviewer A",
            reason="",
        )
        self.assertIn(
            "A reason is required for a completed decision.",
            validate_decision_form(form, self.candidate),
        )

    def test_approval_requires_both_human_source_confirmations(self):
        form = self.form(
            decision="APPROVE",
            reviewer="Reviewer A",
            reason="Checked manually.",
        )
        errors = validate_decision_form(form, self.candidate)
        self.assertIn(
            "APPROVE requires explicit YES confirmation that the answer is "
            "supported by the passage.",
            errors,
        )
        self.assertIn(
            "APPROVE requires explicit YES confirmation that the legal "
            "reference was checked against the original source.",
            errors,
        )
        confirmed = self.form(
            decision="APPROVE",
            reviewer="Reviewer A",
            reason="Checked against the official text.",
            answer_supported_by_passage_confirmation="YES",
            legal_reference_checked_against_original_source="YES",
        )
        self.assertEqual(
            validate_decision_form(confirmed, self.candidate),
            [],
        )

    def test_unconfirmed_reference_requires_exact_reviewer_checked_reference(self):
        candidate = dict(
            self.candidate,
            legal_reference="",
            reference_status="UNCONFIRMED",
        )
        form = {
            "candidate": candidate,
            "review": {
                "decision": "APPROVE",
                "reviewer": "Reviewer A",
                "reason": "Checked source.",
                "answer_supported_by_passage_confirmation": "YES",
                "legal_reference_checked_against_original_source": "YES",
                "edited_answer_rechecked_confirmation": "NO",
                "reviewed_legal_reference": "",
            },
        }
        errors = validate_decision_form(form, candidate)
        self.assertTrue(any("must remain pending" in error for error in errors))
        form["review"]["reviewed_legal_reference"] = "Article 14"
        self.assertEqual(validate_decision_form(form, candidate), [])

    def test_pending_decision_has_no_review_approval_state(self):
        form = self.form(decision="PENDING")
        self.assertEqual(validate_decision_form(form, self.candidate), [])

    def test_edit_and_recheck_requires_new_answer_and_reason(self):
        missing = self.form(
            decision="EDIT_AND_RECHECK",
            reviewer="Reviewer A",
            reason="Needs clearer wording.",
        )
        self.assertIn(
            "EDIT_AND_RECHECK requires a proposed edited_answer.",
            validate_decision_form(missing, self.candidate),
        )
        edited = self.form(
            decision="EDIT_AND_RECHECK",
            reviewer="Reviewer A",
            reason="Rephrase for clarity.",
            edited_answer="The State shall not deny equality before the law.",
        )
        self.assertEqual(validate_decision_form(edited, self.candidate), [])

    def test_approved_record_must_pass_deterministic_checks_too(self):
        form = self.form(
            decision="APPROVE",
            reviewer="Reviewer A",
            reason="Checked the passage and official source.",
            answer_supported_by_passage_confirmation="YES",
            legal_reference_checked_against_original_source="YES",
        )
        page_texts = {
            (
                "constitution_of_india.pdf",
                37,
            ): self.candidate["supporting_passage"],
        }
        self.assertEqual(validate_approved_form(form, page_texts), [])

        unsupported = self.form(
            decision="APPROVE",
            reviewer="Reviewer A",
            reason="Checked the passage and official source.",
            answer_supported_by_passage_confirmation="YES",
            legal_reference_checked_against_original_source="YES",
            edited_answer="A monetary remedy is guaranteed.",
        )
        # An edit must be rechecked in a separate decision, not approved in-place.
        self.assertTrue(validate_decision_form(unsupported, self.candidate))
        self.assertIn(
            "Answer is not found in the supporting passage.",
            validate_approved_form(unsupported, page_texts),
        )

    def test_unconfirmed_reference_cannot_be_approved_without_source_check(self):
        candidate = dict(
            self.candidate,
            legal_reference="",
            reference_status="UNCONFIRMED",
        )
        form = self.form(
            decision="APPROVE",
            reviewer="Reviewer A",
            reason="Reviewed.",
            answer_supported_by_passage_confirmation="YES",
            legal_reference_checked_against_original_source="NO",
        )
        form["candidate"] = candidate
        self.assertTrue(validate_decision_form(form, candidate))

    def test_candidate_snapshot_cannot_be_changed_in_review_form(self):
        changed = dict(self.form_candidate, supporting_passage="Changed evidence.")
        self.assertEqual(
            _candidate_form_mismatches(changed, self.candidate),
            ["supporting_passage"],
        )

    def test_import_keeps_pending_edited_and_incomplete_records_out_of_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            incomplete, incomplete_dir = self.process(
                root,
                {
                    "decision": "APPROVE",
                    "reviewer": "Reviewer A",
                    "reason": "Started review.",
                    "answer_supported_by_passage_confirmation": "YES",
                    "legal_reference_checked_against_original_source": "NO",
                },
                "incomplete",
            )
            self.assertEqual(incomplete["approved_count"], 0)
            self.assertEqual(incomplete["pending_count"], 1)
            self.assertEqual(
                (incomplete_dir / "approved_review_candidates.jsonl")
                .read_text(encoding="utf-8"),
                "",
            )

            pending, pending_dir = self.process(
                root,
                {"decision": "PENDING"},
                "pending",
            )
            self.assertEqual(pending["pending_count"], 1)
            self.assertEqual(
                (pending_dir / "approved_review_candidates.jsonl")
                .read_text(encoding="utf-8"),
                "",
            )

            edited, edited_dir = self.process(
                root,
                {
                    "decision": "EDIT_AND_RECHECK",
                    "reviewer": "Reviewer A",
                    "reason": "Improve wording.",
                    "edited_answer": self.candidate["proposed_answer"],
                },
                "edited",
            )
            self.assertEqual(edited["edited_pending_count"], 1)
            self.assertEqual(edited["approved_count"], 0)
            self.assertEqual(
                (edited_dir / "approved_review_candidates.jsonl")
                .read_text(encoding="utf-8"),
                "",
            )

    def test_imports_approved_record_only_after_confirmations_and_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            summary, output_dir = self.process(
                Path(temp),
                {
                    "decision": "APPROVE",
                    "reviewer": "Reviewer A",
                    "reason": "Checked answer and citation in the source.",
                    "answer_supported_by_passage_confirmation": "YES",
                    "legal_reference_checked_against_original_source": "YES",
                },
                "approved",
            )
            self.assertEqual(summary["approved_count"], 1)
            approved = json.loads(
                (output_dir / "approved_review_candidates.jsonl")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                approved["review_status"],
                "HUMAN_APPROVED_DETERMINISTIC_CHECKS_PASSED",
            )
            self.assertEqual(
                approved["answer_supported_by_passage_confirmation"],
                "YES",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
