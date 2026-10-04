"""Tests for complete, non-destructive Stage 25 dashboard exports."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from io import StringIO
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage25_review_dashboard_data import (
    build_csv_export,
    build_external_review_report,
    build_json_export,
    build_saved_decisions,
    load_dashboard_records,
    write_exports,
    write_saved_decisions,
)


class Stage25ReviewDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_dashboard_records()

    def test_all_21_candidates_are_present_once_and_map_to_full_sources(self):
        ids = [record["candidate_id"] for record in self.records]
        self.assertEqual(len(ids), 21)
        self.assertEqual(len(set(ids)), 21)
        for record in self.records:
            self.assertTrue(record["original_question"])
            self.assertTrue(record["original_answer"])
            self.assertTrue(record["supporting_passage"])
            self.assertTrue(record["source_page_context"])
            self.assertTrue(record["stage16_original_record"])
            self.assertTrue(record["stage22_corrected_record"])
            self.assertTrue(record["stage23_audit_record"])

    def test_json_export_contains_full_candidate_audit_context_and_ids(self):
        decisions = {record["candidate_id"]: "PENDING" for record in self.records}
        data = build_json_export(self.records, decisions)
        self.assertEqual(data["candidate_count"], 21)
        self.assertEqual(
            {record["candidate_id"] for record in data["records"]},
            {record["candidate_id"] for record in self.records},
        )
        first = data["records"][0]
        source = next(
            record for record in self.records
            if record["candidate_id"] == first["candidate_id"]
        )
        self.assertEqual(first["original_question"], source["original_question"])
        self.assertEqual(first["original_answer"], source["original_answer"])
        self.assertEqual(first["supporting_passage"], source["supporting_passage"])
        self.assertEqual(
            first["source_page_context"],
            source["source_page_context"],
        )
        self.assertEqual(
            first["stage23_audit_record"],
            source["stage23_audit_record"],
        )
        self.assertFalse(first["review_state"]["draft_is_saved"])
        self.assertFalse(first["review_state"]["final_dataset_eligible"])
        self.assertIn("current-law status", data["legal_correctness_notice"])

    def test_csv_and_external_report_preserve_untruncated_text_and_candidate_ids(self):
        decisions = {record["candidate_id"]: "PENDING" for record in self.records}
        data = build_json_export(self.records, decisions)
        csv_text = build_csv_export(data)
        csv_records = {
            row["candidate_id"]: row
            for row in csv.DictReader(StringIO(csv_text))
        }
        readable_report = build_external_review_report(self.records, decisions)
        for record in self.records:
            self.assertIn(record["candidate_id"], csv_text)
            self.assertIn(record["candidate_id"], readable_report)
            exported_row = csv_records[record["candidate_id"]]
            self.assertEqual(
                exported_row["original_question"],
                record["original_question"],
            )
            self.assertIn(record["original_question"], readable_report)
            self.assertEqual(
                exported_row["supporting_passage"],
                record["supporting_passage"],
            )
            self.assertIn(record["supporting_passage"], readable_report)
            for page in record["source_page_context"]:
                if page["text"].strip():
                    self.assertIn(page["text"].strip(), readable_report)
        self.assertIn("AI recommendation (not a legal conclusion)", readable_report)
        self.assertIn("not submitted", readable_report)

    def test_draft_selections_are_separate_from_explicitly_saved_decisions(self):
        drafts = {record["candidate_id"]: "PENDING" for record in self.records}
        first_id = self.records[0]["candidate_id"]
        drafts[first_id] = "APPROVE"
        exported = build_json_export(self.records, drafts)
        self.assertEqual(
            exported["records"][0]["review_state"]["draft_selection"],
            "APPROVE",
        )
        self.assertFalse(exported["records"][0]["review_state"]["draft_is_saved"])

        saved = build_saved_decisions(self.records, drafts)
        self.assertEqual(saved["decisions"][0]["decision"], "APPROVE")
        self.assertEqual(
            saved["decisions"][0]["decision_status"],
            "USER_SAVED_SELECTION_NOT_APPROVAL",
        )
        self.assertFalse(saved["decisions"][0]["final_dataset_eligible"])
        self.assertFalse(exported["records"][0]["review_state"]["draft_is_saved"])
        matching_saved_export = build_json_export(
            self.records,
            drafts,
            saved_decisions=drafts,
        )
        changed_drafts = drafts.copy()
        changed_drafts[first_id] = "REJECT"
        changed_export = build_json_export(
            self.records,
            changed_drafts,
            saved_decisions=drafts,
        )
        self.assertTrue(
            matching_saved_export["records"][0]["review_state"]["draft_is_saved"]
        )
        self.assertFalse(changed_export["records"][0]["review_state"]["draft_is_saved"])
        self.assertEqual(
            changed_export["records"][0]["review_state"]["previously_saved_selection"],
            "APPROVE",
        )

    def test_saving_does_not_mutate_source_records_and_writes_a_separate_file(self):
        source_before = deepcopy(self.records)
        decisions = {
            record["candidate_id"]: "EDIT_AND_RECHECK"
            for record in self.records
        }
        payload = build_saved_decisions(self.records, decisions)
        complete_export = build_json_export(
            self.records,
            decisions,
            saved_decisions=decisions,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_saved_decisions(
                Path(directory),
                payload,
                complete_export,
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            full_export = json.loads(
                (path.parent / "complete_review_export.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(len(saved["decisions"]), 21)
        self.assertTrue(all(
            item["decision_status"] == "USER_SAVED_SELECTION_NOT_APPROVAL"
            and item["final_dataset_eligible"] is False
            for item in saved["decisions"]
        ))
        self.assertEqual(self.records, source_before)
        self.assertEqual(full_export["candidate_count"], 21)
        self.assertTrue(all(
            row["review_state"]["draft_is_saved"]
            and not row["review_state"]["final_dataset_eligible"]
            for row in full_export["records"]
        ))
        self.assertEqual(
            full_export["records"][0]["stage16_original_record"],
            self.records[0]["stage16_original_record"],
        )

    def test_export_action_writes_complete_json_and_csv_to_separate_files(self):
        decisions = {
            record["candidate_id"]: "PENDING"
            for record in self.records
        }
        export_data = build_json_export(self.records, decisions)
        json_text = json.dumps(export_data, ensure_ascii=False, indent=2)
        csv_text = build_csv_export(export_data)
        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = write_exports(
                Path(directory),
                json_text,
                csv_text,
            )
            saved_json = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                saved_csv = stream.read()
        self.assertEqual(saved_json["candidate_count"], 21)
        self.assertEqual(saved_csv, csv_text)
        self.assertEqual(json_path.parent, csv_path.parent)
        self.assertNotEqual(json_path, csv_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
