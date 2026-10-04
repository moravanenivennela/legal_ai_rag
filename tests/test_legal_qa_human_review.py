"""Deterministic tests for the isolated human QA review workflow."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_source_grounded_qa_candidates import _make_passage, generate_passage_candidates
from scripts.legal_qa_human_review import (
    EDITED_PENDING_STATUS,
    REVIEWED_STATUS,
    _validate_review_decision,
    group_review_candidates,
    import_review_decisions,
    validate_reviewed_answer,
)


class LegalQaHumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.evidence = (
            "14. Equality before law. The State shall not deny to any person "
            "equality before the law or the equal protection of the laws."
        )
        passage = _make_passage("constitution_of_india.pdf", 37, self.evidence)
        self.candidate = generate_passage_candidates(passage)[0]
        self.candidate["candidate_id"] = "candidate-1"

    def test_completed_decisions_require_reviewer_and_reason(self):
        self.assertEqual(
            _validate_review_decision({
                "review_decision": "APPROVE",
                "reviewer": "",
                "reason": "Reviewed",
            }),
            ["reviewer is required for a completed decision."],
        )
        self.assertEqual(
            _validate_review_decision({
                "review_decision": "APPROVE",
                "reviewer": "Reviewer A",
                "reason": "",
            }),
            ["reason is required for a completed decision."],
        )
        self.assertEqual(
            _validate_review_decision({
                "review_decision": "NOT_A_DECISION",
                "reviewer": "Reviewer A",
                "reason": "Reviewed",
            }),
            ["Unsupported review decision: NOT_A_DECISION."],
        )

    def test_edited_answer_is_revalidated_against_evidence(self):
        page_texts = {
            ("constitution_of_india.pdf", 37): self.evidence,
        }
        self.assertEqual(
            validate_reviewed_answer(
                self.candidate,
                "The State shall not deny to any person equality before the law.",
                page_texts,
            ),
            [],
        )
        errors = validate_reviewed_answer(
            self.candidate,
            "The State must provide a monetary remedy.",
            page_texts,
        )
        self.assertIn(
            "Answer is not an extract from the linked evidence.",
            errors,
        )

    def test_duplicate_questions_are_grouped_for_joint_review(self):
        duplicate = dict(self.candidate, candidate_id="candidate-2")
        rows = group_review_candidates([self.candidate, duplicate])
        self.assertEqual(rows[0]["duplicate_group_id"], rows[1]["duplicate_group_id"])
        self.assertTrue(rows[0]["duplicate_group_id"])
        self.assertEqual(rows[0]["duplicate_candidate_ids"], "candidate-2")
        self.assertIn("EXACT_QUESTION_DUPLICATE", rows[0]["duplicate_flags"])

    def test_source_reference_conflict_links_related_candidates(self):
        conflicting = dict(
            self.candidate,
            candidate_id="candidate-conflict",
            legal_reference="Section 14",
        )
        rows = group_review_candidates([self.candidate, conflicting])
        conflict_row = next(
            row for row in rows
            if row["candidate_id"] == "candidate-conflict"
        )
        self.assertTrue(conflict_row["source_reference_conflict"])
        self.assertIn(
            "candidate-1",
            conflict_row["source_reference_conflict_candidate_ids"],
        )
        self.assertEqual(
            conflict_row["source_reference_conflict_group_id"],
            rows[0]["source_reference_conflict_group_id"],
        )

    def test_only_approved_and_validated_rows_enter_reviewed_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            source_dir = Path(temp) / "input"
            source_dir.mkdir()
            with (source_dir / "candidates.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(self.candidate) + "\n")

            worksheet = Path(temp) / "review.csv"
            columns = [
                "candidate_id",
                "review_decision",
                "reviewer",
                "reason",
                "reviewed_answer",
            ]
            with worksheet.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow({
                    "candidate_id": "candidate-1",
                    "review_decision": "",
                    "reviewer": "",
                    "reason": "",
                    "reviewed_answer": "",
                })
            pending_summary = import_review_decisions(
                source_dir,
                worksheet,
                Path(temp) / "pending-import",
                {("constitution_of_india.pdf", 37): self.evidence},
            )
            self.assertEqual(pending_summary["still_pending_count"], 1)
            self.assertEqual(pending_summary["eligible_reviewed_dataset_count"], 0)
            self.assertEqual(
                (Path(temp) / "pending-import" / "reviewed_qa.jsonl")
                .read_text(encoding="utf-8"),
                "",
            )

            with worksheet.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow({
                    "candidate_id": "candidate-1",
                    "review_decision": "APPROVE",
                    "reviewer": "Reviewer A",
                    "reason": "Manually reviewed",
                    "reviewed_answer": "",
                })
            approved_summary = import_review_decisions(
                source_dir,
                worksheet,
                Path(temp) / "approved-import",
                {("constitution_of_india.pdf", 37): self.evidence},
            )
            self.assertEqual(approved_summary["approved_count"], 1)
            self.assertEqual(approved_summary["eligible_reviewed_dataset_count"], 1)
            approved = json.loads(
                (Path(temp) / "approved-import" / "reviewed_qa.jsonl")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(approved["review_status"], REVIEWED_STATUS)

    def test_edit_and_recheck_stays_out_of_approved_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            source_dir = Path(temp) / "input"
            source_dir.mkdir()
            (source_dir / "candidates.jsonl").write_text(
                json.dumps(self.candidate) + "\n",
                encoding="utf-8",
            )
            worksheet = Path(temp) / "review.csv"
            with worksheet.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "candidate_id", "review_decision", "reviewer",
                        "reason", "reviewed_answer",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "candidate_id": "candidate-1",
                    "review_decision": "EDIT_AND_RECHECK",
                    "reviewer": "Reviewer A",
                    "reason": "Clarified wording",
                    "reviewed_answer": (
                        "The State shall not deny to any person equality before the law."
                    ),
                })
            summary = import_review_decisions(
                source_dir,
                worksheet,
                Path(temp) / "edited-import",
                {("constitution_of_india.pdf", 37): self.evidence},
            )
            self.assertEqual(summary["edited_count"], 1)
            self.assertEqual(summary["eligible_reviewed_dataset_count"], 0)
            imported = json.loads(
                (Path(temp) / "edited-import" / "review_import_results.jsonl")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(imported["review_status"], EDITED_PENDING_STATUS)

    def test_unsupported_approved_answer_is_rejected_from_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            source_dir = Path(temp) / "input"
            source_dir.mkdir()
            (source_dir / "candidates.jsonl").write_text(
                json.dumps(self.candidate) + "\n",
                encoding="utf-8",
            )
            worksheet = Path(temp) / "review.csv"
            with worksheet.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "candidate_id", "review_decision", "reviewer",
                        "reason", "reviewed_answer",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "candidate_id": "candidate-1",
                    "review_decision": "APPROVE",
                    "reviewer": "Reviewer A",
                    "reason": "Reviewed",
                    "reviewed_answer": "A monetary remedy is guaranteed.",
                })
            output_dir = Path(temp) / "invalid-import"
            summary = import_review_decisions(
                source_dir,
                worksheet,
                output_dir,
                {("constitution_of_india.pdf", 37): self.evidence},
            )
            self.assertEqual(summary["approved_count"], 0)
            self.assertEqual(summary["validation_rejected_count"], 1)
            self.assertEqual(summary["eligible_reviewed_dataset_count"], 0)
            self.assertEqual(
                (output_dir / "reviewed_qa.jsonl").read_text(encoding="utf-8"),
                "",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
