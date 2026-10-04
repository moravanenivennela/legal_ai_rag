"""Focused tests for the non-destructive full legal QA candidate audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.legal_qa_audit_core import (
    _refresh_suggestion_status,
    audit_candidate,
    detect_duplicate_groups,
    discover_candidate_files,
    run_full_audit,
)


class FullLegalQaAuditTests(unittest.TestCase):
    def setUp(self):
        self.pages = {
            "constitution_of_india.pdf": [
                "14. Equality before law.\n"
                "The State shall not deny to any person equality before the law."
            ],
            "consumer_protection_act_2019.pdf": [
                "2. In this Act, unless the context otherwise requires,\n"
                "(1) definition."
            ],
        }
        self.candidate = {
            "candidate_id": "candidate-a",
            "question": "What does Article 14 state?",
            "answer": "The State shall not deny to any person equality before the law.",
            "reference": "Article 14",
            "source_document": "constitution_of_india.pdf",
            "page": "1",
            "supporting_passage": (
                "14. Equality before law.\n"
                "The State shall not deny to any person equality before the law."
            ),
            "source_occurrences": [],
            "source_record_count": 1,
            "source_field_variants": {},
            "prior_ai_recommendations": [],
            "prior_decisions": [],
            "input_errors": [],
        }

    def audit(self, candidate=None, pages=None, errors=None):
        return audit_candidate(
            candidate or self.candidate,
            pages if pages is not None else self.pages,
            errors or {},
        )

    def test_exact_source_passage_and_reference_are_only_source_supported(self):
        result = self.audit()
        self.assertEqual(result["automated_status"], "SUPPORTED_CANDIDATE")
        self.assertEqual(result["exact_source_evidence_excerpt"],
                         self.candidate["supporting_passage"])
        self.assertTrue(result["answer_exactly_supported_by_supplied_passage"])
        self.assertEqual(result["human_approval_status"], "NOT_SET_BY_AUTOMATION")
        self.assertFalse(result["current_law_status"] != "NOT_CHECKED")

    def test_citation_mismatch_is_flagged_against_pdf_heading_pages(self):
        pages = {
            **self.pages,
            "constitution_of_india.pdf": [
                "14. Equality before law.",
                "15. Prohibition of discrimination.",
            ],
        }
        candidate = {**self.candidate, "reference": "Article 15", "page": "1"}
        result = self.audit(candidate, pages)
        self.assertEqual(result["automated_status"], "UNSUPPORTED_OR_MISMATCHED")
        self.assertEqual(result["reference_check_status"], "REFERENCE_PAGE_MISMATCH")

    def test_answer_number_not_supported_by_passage_is_mismatched(self):
        candidate = {
            **self.candidate,
            "answer": "The State shall not deny equality after 999 years.",
        }
        result = self.audit(candidate)
        self.assertEqual(result["automated_status"], "UNSUPPORTED_OR_MISMATCHED")
        self.assertIn("ANSWER_NUMERIC_OR_DATE_NOT_IN_EVIDENCE", result["issues"])

    def test_duplicate_group_keeps_originals_and_prefers_stronger_provenance(self):
        first = {
            "candidate_id": "candidate-a",
            "original_question": "What does Article 14 state about equality?",
            "original_answer": "Equality before law.",
            "automated_status": "REQUIRES_HUMAN_LEGAL_REVIEW",
            "exact_source_evidence_excerpt": "",
            "reference_check_status": "REFERENCE_UNRESOLVED",
        }
        second = {
            "candidate_id": "candidate-b",
            "original_question": "What does Article 14 state about equality?",
            "original_answer": "The State shall not deny equality before the law.",
            "automated_status": "SUPPORTED_CANDIDATE",
            "exact_source_evidence_excerpt": "Article 14 source text",
            "reference_check_status": "REFERENCE_MATCHED_PDF_HEADING",
        }
        records = [first, second]
        groups = detect_duplicate_groups(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["preferred_candidate_id"], "candidate-b")
        self.assertEqual(records[0]["automated_status"], "DUPLICATE")
        self.assertEqual(records[1]["automated_status"], "SUPPORTED_CANDIDATE")
        self.assertEqual(len(groups[0]["members"]), 2)

    def test_near_duplicate_question_is_grouped_without_overwriting_text(self):
        questions = [
            "Which protection is described by Article 14 of the Constitution?",
            "What protection is described by Article 14 of the Constitution?",
        ]
        records = [
            {
                "candidate_id": f"near-{index}",
                "original_question": question,
                "original_answer": f"Original answer {index}",
                "automated_status": "REQUIRES_HUMAN_LEGAL_REVIEW",
                "exact_source_evidence_excerpt": "",
                "reference_check_status": "REFERENCE_UNRESOLVED",
            }
            for index, question in enumerate(questions)
        ]
        original_questions = [row["original_question"] for row in records]
        groups = detect_duplicate_groups(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["duplicate_relation"],
            "NEAR_QUESTION",
        )
        self.assertEqual(
            [row["original_question"] for row in records],
            original_questions,
        )

    def test_ocr_punctuation_artifact_gets_source_backed_suggestion_only(self):
        candidate = {
            **self.candidate,
            "answer": (
                "The State shall not deny to any person equality?before the law."
            ),
            "supporting_passage": (
                "14. Equality before law.\n"
                "The State shall not deny to any person equality before the law."
            ),
        }
        result = self.audit(candidate)
        self.assertEqual(result["automated_status"], "NEEDS_CORRECTION")
        self.assertIn("POSSIBLE_OCR_PUNCTUATION_ARTIFACT", result["issues"])
        self.assertEqual(
            result["suggested_answer"],
            "The State shall not deny to any person equality before the law.",
        )
        self.assertEqual(result["suggestion_status"], "SUGGESTED_ONLY_NOT_APPROVED")

    def test_missing_pdf_is_insufficient_not_silently_assumed(self):
        pages = {
            "constitution_of_india.pdf": None,
            "consumer_protection_act_2019.pdf": self.pages[
                "consumer_protection_act_2019.pdf"
            ],
        }
        result = self.audit(
            pages=pages,
            errors={"constitution_of_india.pdf": "FileNotFoundError: missing"},
        )
        self.assertEqual(
            result["automated_status"],
            "INSUFFICIENT_SOURCE_EVIDENCE",
        )
        self.assertIn("SOURCE_PDF_MISSING_OR_UNREADABLE", result["issues"])

    def test_paraphrased_answer_is_kept_for_human_legal_review(self):
        candidate = {
            **self.candidate,
            "answer": "The government must not refuse anyone equal treatment by law.",
        }
        result = self.audit(candidate)
        self.assertEqual(
            result["automated_status"],
            "REQUIRES_HUMAN_LEGAL_REVIEW",
        )
        self.assertIn(
            "ANSWER_IS_PARAPHRASE_REQUIRING_SEMANTIC_REVIEW",
            result["issues"],
        )

    def test_schedule_paragraph_reference_is_not_mistaken_for_article_number(self):
        pages = {
            "constitution_of_india.pdf": [
                "1. TENTH SCHEDULE\n"
                "1. Interpretation.\n"
                "(d) \"paragraph\" means a paragraph of this Schedule."
            ],
            "consumer_protection_act_2019.pdf": self.pages[
                "consumer_protection_act_2019.pdf"
            ],
        }
        candidate = {
            **self.candidate,
            "question": "What does paragraph 1(d) define?",
            "answer": "\"paragraph\" means a paragraph of this Schedule.",
            "reference": "Tenth Schedule, paragraph 1(d)",
            "page": "1",
            "supporting_passage": pages["constitution_of_india.pdf"][0],
        }
        result = self.audit(candidate, pages)
        self.assertEqual(
            result["reference_check_status"],
            "REFERENCE_MATCHED_PDF_HEADING",
        )
        self.assertNotEqual(
            result["automated_status"],
            "UNSUPPORTED_OR_MISMATCHED",
        )

    def test_suggestion_count_ignores_whitespace_but_flags_real_punctuation_change(self):
        record = {
            "original_question": "What does the provision state?",
            "original_answer": "Equality\nbefore the law.",
            "original_reference": "Article 14",
            "suggested_question": "",
            "suggested_answer": "Equality before the law.",
            "suggested_reference": "",
            "source_matched_answer_excerpt": "Equality before the law.",
        }
        _refresh_suggestion_status(record)
        self.assertEqual(record["suggested_fields_changed"], [])
        self.assertFalse(record["source_supported_correction_suggested"])

        record["suggested_answer"] = "Equality, before the law."
        record["source_matched_answer_excerpt"] = "Equality, before the law."
        _refresh_suggestion_status(record)
        self.assertEqual(record["suggested_fields_changed"], ["answer"])
        self.assertTrue(record["source_supported_correction_suggested"])

    def test_candidate_discovery_excludes_domain_classifier_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classification = (
                root
                / "fine_tuning"
                / "dataset"
                / "expanded"
                / "domain_guardrail_500"
            )
            evaluation = (
                root
                / "fine_tuning"
                / "dataset"
                / "expanded"
                / "evaluation_500_new"
            )
            classification.mkdir(parents=True)
            evaluation.mkdir(parents=True)
            classifier_path = (
                classification / "domain_guardrail_500_candidate_questions.csv"
            )
            qa_path = evaluation / "legal_evaluation_500_candidates.csv"
            classifier_path.touch()
            qa_path.touch()
            discovered = set(discover_candidate_files(root))
        self.assertIn(qa_path, discovered)
        self.assertNotIn(classifier_path, discovered)

    def test_report_run_writes_required_files_and_preserves_source_inputs(self):
        if __import__("scripts.legal_qa_audit_core", fromlist=["pymupdf"]).pymupdf is None:
            self.skipTest("PyMuPDF is required to create the local fixture PDF.")
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports" / (
                "qa_candidate_generation_stage13_20261004T091039Z_seed42"
            )
            reports.mkdir(parents=True)
            data = root / "data"
            data.mkdir()
            pdf_path = data / "constitution_of_india.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "14. Equality before law.\n"
                "The State shall not deny to any person equality before the law.",
            )
            document.save(pdf_path)
            document.close()
            candidate_path = reports / "candidates.jsonl"
            candidate_row = {
                **self.candidate,
                "source_filename": "constitution_of_india.pdf",
                "source": None,
            }
            candidate_path.write_text(
                json.dumps(candidate_row) + "\n",
                encoding="utf-8",
            )
            original_candidate = candidate_path.read_bytes()
            output_dir = run_full_audit(
                root=root,
                data_dir=data,
                output_root=root / "reports",
                batch_size=1,
            )
            required = {
                "full_audit.json",
                "full_audit.csv",
                "summary_report.md",
                "supported_candidates.json",
                "needs_correction.json",
                "unsupported_candidates.json",
                "duplicates.json",
                "human_review_required.json",
                "audit_metrics.json",
            }
            self.assertTrue(required.issubset({p.name for p in output_dir.iterdir()}))
            self.assertEqual(candidate_path.read_bytes(), original_candidate)
            report = json.loads(
                (output_dir / "full_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["count_unique_candidates"], 1)
            metrics = json.loads(
                (output_dir / "audit_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["human_approved_count_by_this_automation"], 0)
            self.assertEqual(metrics["accuracy_precision_recall_f1"], "NOT_MEASURED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
