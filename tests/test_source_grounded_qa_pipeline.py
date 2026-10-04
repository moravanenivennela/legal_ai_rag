"""Deterministic tests for the isolated source-grounded QA pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_source_grounded_qa_candidates import (
    HUMAN_REVIEW_STATUS,
    _make_passage,
    build_human_review_rows,
    deduplicate_candidates,
    extract_legal_reference,
    extract_pilot_passages,
    generate_passage_candidates,
    generate_from_passages,
    reference_supported,
    select_balanced_candidates,
    select_source_balanced_candidates,
    split_page_into_passages,
    validate_candidate,
    validate_split_leakage,
    write_run_outputs,
)


class SourceGroundedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.article_passage = _make_passage(
            "constitution_of_india.pdf",
            37,
            (
                "14. Equality before law. The State shall not deny to any person "
                "equality before the law or the equal protection of the laws "
                "within the territory of India."
            ),
        )
        self.consumer_passage = _make_passage(
            "consumer_protection_act_2019.pdf",
            2,
            (
                "2. In this Act, unless the context otherwise requires, "
                '(1) "advertisement" means any audio or visual publicity, '
                "representation, endorsement or pronouncement made by means "
                "of light, sound, smoke, gas, print, electronic media, internet "
                "or website and includes any notice, circular, label, wrapper, "
                "invoice or such other documents;"
            ),
        )

    def test_reference_extraction_requires_explicit_numbered_context(self):
        self.assertEqual(
            extract_legal_reference(
                "constitution_of_india.pdf",
                "14. Equality before law. The State shall not deny equality.",
            ),
            ("Article 14", "EXPLICIT_NUMBERED_HEADING"),
        )
        self.assertEqual(
            extract_legal_reference(
                "consumer_protection_act_2019.pdf",
                '2. In this Act, unless the context otherwise requires, '
                '(1) "advertisement" means any visual publicity.',
            ),
            ("Section 2(1)", "EXPLICIT_NUMBERED_HEADING"),
        )
        self.assertEqual(
            extract_legal_reference(
                "constitution_of_india.pdf",
                "Contents: 14. Equality before law.",
            ),
            (None, "UNCONFIRMED"),
        )
        self.assertEqual(
            extract_legal_reference("constitution_of_india.pdf", "Equality before law."),
            (None, "UNCONFIRMED"),
        )
        section_evidence = (
            "2. In this Act, unless the context otherwise requires,— "
            '(1) "advertisement" means any publicity; '
            '(2) "goods" means items for sale;'
        )
        self.assertTrue(
            reference_supported(
                "consumer_protection_act_2019.pdf",
                section_evidence,
                "Section 2(2)",
            )
        )
        self.assertFalse(
            reference_supported(
                "consumer_protection_act_2019.pdf",
                section_evidence,
                "Section 2(3)",
            )
        )

    def test_small_source_fixtures_extract_one_passage_per_document(self):
        passages = extract_pilot_passages({
            "constitution_of_india.pdf": [
                "unused" for _ in range(36)
            ] + [
                "14. Equality before law. The State shall not deny equality.\n"
                "15. Prohibition of discrimination.",
            ],
            "consumer_protection_act_2019.pdf": [
                "unused",
                (
                    "Definitions. 2. In this Act, unless the context otherwise "
                    'requires, (1) "advertisement" means any audio publicity; '
                    '(2) "goods" means items for sale.'
                ),
            ],
        })
        self.assertEqual(
            [(item["source_filename"], item["page"]) for item in passages],
            [
                ("constitution_of_india.pdf", 37),
                ("consumer_protection_act_2019.pdf", 2),
            ],
        )
        self.assertNotIn("15. Prohibition", passages[0]["evidence"])
        self.assertNotIn('(2) "goods"', passages[1]["evidence"])
        self.assertEqual(passages[0]["legal_reference"], "Article 14")
        self.assertEqual(passages[1]["legal_reference"], "Section 2(1)")

    def test_templates_are_source_extracts_and_candidates_need_review(self):
        article_candidates = generate_passage_candidates(self.article_passage)
        consumer_candidates = generate_passage_candidates(self.consumer_passage)
        self.assertTrue(article_candidates)
        self.assertTrue(consumer_candidates)
        article_direct = next(
            row for row in article_candidates
            if row["question_type"] == "direct_factual"
        )
        definition = next(
            row for row in consumer_candidates
            if row["question_type"] == "definition"
        )
        self.assertTrue(article_direct["answer"].startswith("The State shall"))
        self.assertIn("by means of light", definition["answer"])
        self.assertIn("or website", definition["answer"])
        self.assertNotIn("(b) ", definition["answer"])
        self.assertTrue(definition["answer"].endswith(";"))
        for candidate in article_candidates + consumer_candidates:
            self.assertEqual(candidate["review_status"], HUMAN_REVIEW_STATUS)
            self.assertIn(candidate["answer"], candidate["evidence"])
            self.assertEqual(
                candidate["source_filename"],
                self.article_passage["source_filename"]
                if candidate in article_candidates
                else self.consumer_passage["source_filename"],
            )
        all_candidates = article_candidates + consumer_candidates
        self.assertEqual(
            len({row["candidate_id"] for row in all_candidates}),
            len(all_candidates),
        )

    def test_repeated_template_candidates_keep_distinct_stable_ids(self):
        passage = _make_passage(
            "constitution_of_india.pdf",
            50,
            (
                "14. Equality before law.—The State shall not deny equality. "
                "Provided that the first stated condition applies; "
                "Provided that the second stated condition applies."
            ),
        )
        candidates = generate_passage_candidates(passage)
        identifiers = [row["candidate_id"] for row in candidates]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_candidate_schema_page_reference_and_evidence_validation(self):
        candidates = (
            generate_passage_candidates(self.article_passage)
            + generate_passage_candidates(self.consumer_passage)
        )
        pages = {
            (
                self.article_passage["source_filename"],
                self.article_passage["page"],
            ): self.article_passage["evidence"],
            (
                self.consumer_passage["source_filename"],
                self.consumer_passage["page"],
            ): self.consumer_passage["evidence"],
        }
        for candidate in candidates:
            self.assertEqual(validate_candidate(candidate, pages), [])

        broken = dict(candidates[0], page=900)
        self.assertIn(
            "Source/page pair is not present in extracted source text.",
            validate_candidate(broken, pages),
        )
        broken = dict(candidates[0], legal_reference="Article 14 and equality")
        self.assertIn(
            "Legal reference does not match the supported format.",
            validate_candidate(broken, pages),
        )
        broken = dict(candidates[0], answer="An unsupported legal conclusion.")
        self.assertIn(
            "Answer is not an extract from the linked evidence.",
            validate_candidate(broken, pages),
        )

    def test_deduplication_keeps_distinct_questions_with_shared_answer(self):
        base = generate_passage_candidates(self.article_passage)[0]
        first = dict(
            base,
            candidate_id="first",
            question="What obligation is imposed on the State?",
            answer="The State shall not deny equality before the law.",
        )
        second = dict(
            base,
            candidate_id="second",
            question="Which legal protection is stated for each person?",
            answer=first["answer"],
        )
        exact_duplicate = dict(first, candidate_id="exact-duplicate")

        retained, rejected = deduplicate_candidates(
            [first, second, exact_duplicate]
        )

        self.assertEqual(
            {row["candidate_id"] for row in retained},
            {"first", "second"},
        )
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "EXACT_QUESTION_DUPLICATE")
        self.assertIn("REPEATED_EXACT_ANSWER", first["duplicate_flags"])
        self.assertIn("REPEATED_ANSWER_TEMPLATE", second["duplicate_flags"])

        near_first = dict(
            first,
            candidate_id="near-first",
            question="What duty does this provision state?",
        )
        near_second = dict(
            first,
            candidate_id="near-second",
            question="What duty does that provision state?",
        )
        _, near_rejected = deduplicate_candidates([near_first, near_second])
        self.assertEqual(len(near_rejected), 1)
        self.assertEqual(
            near_rejected[0]["reason"],
            "NEAR_QUESTION_DUPLICATE",
        )

    def test_split_validator_detects_passage_question_and_answer_leakage(self):
        candidate = generate_passage_candidates(self.article_passage)[0]
        duplicate = dict(
            candidate,
            candidate_id="second",
            question="What does Article 14 state about equality before law?",
        )
        errors = validate_split_leakage({
            "train": [candidate],
            "test": [duplicate],
        })
        self.assertTrue(any("Passage leakage" in error for error in errors))
        self.assertTrue(any("Question leakage" in error for error in errors))
        self.assertTrue(any("Answer-template leakage" in error for error in errors))

    def test_source_selection_caps_oversampled_document_deterministically(self):
        rows = [
            {"candidate_id": f"constitution-{i}", "source_filename": "constitution_of_india.pdf"}
            for i in range(4)
        ] + [
            {"candidate_id": "consumer-1", "source_filename": "consumer_protection_act_2019.pdf"}
        ]
        selected = select_source_balanced_candidates(rows, per_source_limit=1, seed=42)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {row["source_filename"] for row in selected},
            {"constitution_of_india.pdf", "consumer_protection_act_2019.pdf"},
        )
        self.assertEqual(
            selected,
            select_source_balanced_candidates(rows, per_source_limit=1, seed=42),
        )

    def test_page_segmentation_preserves_explicit_provision_boundaries(self):
        page_text = (
            "Part III\n"
            "14. Equality before law.—The State shall not deny equality.\n"
            "15. Prohibition of discrimination.—The State shall not discriminate."
        )
        passages = split_page_into_passages(
            "constitution_of_india.pdf",
            37,
            page_text,
        )
        self.assertEqual(len(passages), 2)
        self.assertEqual([row["legal_reference"] for row in passages], [
            "Article 14",
            "Article 15",
        ])
        self.assertEqual(passages[0]["page"], 37)
        self.assertIn("Equality before law", passages[0]["evidence"])
        self.assertNotIn("15. Prohibition", passages[0]["evidence"])

    def test_bounded_generation_validates_then_deduplicates_and_limits(self):
        passages = [self.article_passage, self.consumer_passage]
        page_texts = {
            (row["source_filename"], row["page"]): row["evidence"]
            for row in passages
        }
        results = generate_from_passages(
            passages,
            page_texts,
            candidate_limit=1,
            seed=7,
        )
        self.assertEqual(len(results["raw_candidates"]), 3)
        self.assertEqual(len(results["invalid_candidates"]), 0)
        self.assertEqual(len(results["duplicate_rejections"]), 0)
        self.assertEqual(len(results["selected"]), 1)
        self.assertEqual(results["omitted_by_candidate_limit"], 2)
        self.assertEqual(
            results["selected"][0]["review_status"],
            HUMAN_REVIEW_STATUS,
        )

    def test_candidate_limit_and_seed_are_deterministic_and_noninflating(self):
        candidates = (
            generate_passage_candidates(self.article_passage)
            + generate_passage_candidates(self.consumer_passage)
        )
        first = select_balanced_candidates(candidates, 2, seed=13)
        second = select_balanced_candidates(candidates, 2, seed=13)
        self.assertEqual(
            [row["candidate_id"] for row in first],
            [row["candidate_id"] for row in second],
        )
        self.assertLessEqual(len(first), 2)
        self.assertEqual(
            len(select_balanced_candidates(candidates, 20, seed=13)),
            3,
        )
        undersupplied = [
            dict(candidates[0], candidate_id=f"candidate-{index}")
            for index in range(3)
        ]
        self.assertEqual(
            len(select_balanced_candidates(undersupplied, 10, seed=13)),
            len(undersupplied),
        )
        skewed = [
            dict(
                candidates[index % len(candidates)],
                candidate_id=f"constitution-{index}",
                source_filename="constitution_of_india.pdf",
            )
            for index in range(20)
        ] + [
            dict(
                candidates[-1],
                candidate_id="consumer-only",
                source_filename="consumer_protection_act_2019.pdf",
            )
        ]
        balanced = select_balanced_candidates(skewed, 10, seed=13)
        counts = {}
        for row in balanced:
            counts[row["source_filename"]] = counts.get(row["source_filename"], 0) + 1
        self.assertEqual(counts.get("constitution_of_india.pdf", 0), 9)
        self.assertEqual(counts.get("consumer_protection_act_2019.pdf", 0), 1)

    def test_passages_without_supported_patterns_are_classified(self):
        passage = _make_passage(
            "constitution_of_india.pdf",
            38,
            "A fragment that contains no recognized legal question pattern.",
        )
        result = generate_from_passages(
            [passage],
            {
                (passage["source_filename"], passage["page"]): passage["evidence"],
            },
            candidate_limit=20,
            seed=42,
        )
        self.assertEqual(result["raw_candidates"], [])
        self.assertEqual(
            result["no_candidate_passages"][0]["reason"],
            "NO_SUPPORTED_QUESTION_PATTERN",
        )
        self.assertEqual(
            result["no_candidate_passages"][0]["evidence"],
            passage["evidence"],
        )

    def test_run_outputs_are_separate_and_keep_review_fields(self):
        raw_candidates = (
            generate_passage_candidates(self.article_passage)
            + generate_passage_candidates(self.consumer_passage)
        )
        candidates = deduplicate_candidates(raw_candidates)[0]
        raw_candidates[0].setdefault("duplicate_flags", []).append(
            "TEST_DUPLICATE_FLAG"
        )
        raw_candidates[0].setdefault("selection_flags", []).append(
            "TEST_SELECTION_FLAG"
        )
        review_rows = build_human_review_rows(raw_candidates)
        self.assertEqual(len(review_rows), len(raw_candidates))
        self.assertIn("evidence", review_rows[0])
        self.assertIn("validation_flags", review_rows[0])
        self.assertIn("duplicate_flags", review_rows[0])
        self.assertIn("selection_flags", review_rows[0])
        self.assertEqual(
            {row["review_status"] for row in review_rows},
            {HUMAN_REVIEW_STATUS},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            manifest = write_run_outputs(
                output_dir,
                candidates,
                [],
                {"seed": 42, "random_sampling": False},
                review_candidates=raw_candidates,
                no_candidate_passages=[{
                    "passage_id": "no-candidate",
                    "source_filename": "constitution_of_india.pdf",
                    "page": 38,
                    "legal_reference": "",
                    "reason": "NO_SUPPORTED_QUESTION_PATTERN",
                    "evidence": "A fragment.",
                }],
            )
            self.assertEqual(manifest["counts"]["human_review_pending"], len(raw_candidates))
            self.assertFalse(manifest["ollama_used"])
            self.assertTrue((output_dir / "candidates.jsonl").is_file())
            self.assertTrue((output_dir / "human_review_report.csv").is_file())
            self.assertTrue((output_dir / "no_candidate_passages.csv").is_file())
            self.assertEqual(manifest["no_candidate_passages"]["count"], 1)
            self.assertEqual(
                manifest["candidate_quality_flags"]["duplicate_categories"][
                    "TEST_DUPLICATE_FLAG"
                ],
                1,
            )
            self.assertEqual(
                manifest["candidate_quality_flags"]["selection_reasons"][
                    "TEST_SELECTION_FLAG"
                ],
                1,
            )
            self.assertTrue((output_dir / "run_manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                write_run_outputs(output_dir, candidates, [], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
