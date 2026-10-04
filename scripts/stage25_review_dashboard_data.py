"""Load and export Stage 25 review dashboard data without editing source records."""

from __future__ import annotations

import copy
import csv
from io import StringIO
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.guided_stage23_legal_qa_review import (  # noqa: E402
    ALLOWED_SOURCES,
    DEFAULT_AUDIT_CSV,
    DEFAULT_FORMS_DIR,
    DEFAULT_STAGE22_CSV,
    load_review_candidates,
    read_csv,
    read_pdf_context,
)

DECISIONS = ("PENDING", "APPROVE", "REJECT", "EDIT_AND_RECHECK")
EXPORT_NOTICE = (
    "Source-text matching and AI recommendations do not establish legal "
    "correctness or current-law status. Saved dropdown decisions are reviewer "
    "selections only; they are not eligible for a final dataset because this "
    "dashboard does not collect the required reviewer reason or source "
    "confirmations."
)
CSV_FIELDS = (
    "candidate_id",
    "draft_decision",
    "saved_decision",
    "decision_record_status",
    "final_dataset_eligible",
    "original_question",
    "original_answer",
    "suggested_question",
    "suggested_answer",
    "original_reference",
    "proposed_reference",
    "source_pdf",
    "pdf_page_1_based",
    "supporting_passage",
    "source_page_context_json",
    "stage23_audit_concerns",
    "duplicate_flags",
    "stage19_ai_recommendation",
    "stage16_original_record_json",
    "stage22_corrected_record_json",
    "stage23_audit_record_json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_dashboard_records(
    audit_csv: Path = DEFAULT_AUDIT_CSV,
    stage22_csv: Path = DEFAULT_STAGE22_CSV,
    forms_dir: Path = DEFAULT_FORMS_DIR,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Join Stage 16/22/23 facts and extracted PDF context by immutable ID."""
    root_data_dir = data_dir or ROOT / "data"
    page_counts: dict[str, int] = {}
    pdf_paths: dict[str, Path] = {}
    for filename in ALLOWED_SOURCES:
        pdf_path = root_data_dir / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Source PDF not found: {pdf_path}")
        with pymupdf.open(pdf_path) as document:
            page_counts[filename] = document.page_count
        pdf_paths[filename] = pdf_path

    core_records = load_review_candidates(
        audit_csv,
        stage22_csv,
        forms_dir,
        page_counts,
    )
    stage22_by_id = {
        row["candidate_id"]: row
        for row in read_csv(stage22_csv)
        if row.get("priority_group") == "READY_FOR_HUMAN_REVIEW"
    }
    audit_by_id = {
        row["candidate_id"]: row for row in read_csv(audit_csv)
    }
    records: list[dict[str, Any]] = []
    for core in core_records:
        candidate_id = core["candidate_id"]
        form_path = forms_dir / f"{candidate_id}.json"
        form = json.loads(form_path.read_text(encoding="utf-8"))
        source_name = core["source_document"]
        page_context = read_pdf_context(
            pdf_paths[source_name],
            core["pdf_page_1_based"],
        )
        record = copy.deepcopy(core)
        record.update({
            "supporting_passage": form["candidate"].get("supporting_passage", ""),
            "reference_status": form["candidate"].get("reference_status", ""),
            "reference_confidence_flags": form["candidate"].get(
                "reference_confidence_flags", ""
            ),
            "evidence_status": form["candidate"].get("evidence_status", ""),
            "question_type": form["candidate"].get("question_type", ""),
            "source_page_context": [
                {"pdf_page_1_based": page_number, "text": text}
                for page_number, text in page_context
            ],
            "stage16_original_record": copy.deepcopy(form["candidate"]),
            "stage22_corrected_record": copy.deepcopy(stage22_by_id[candidate_id]),
            "stage23_audit_record": copy.deepcopy(audit_by_id[candidate_id]),
            "stage22_reference_check_status": stage22_by_id[candidate_id].get(
                "reference_check_status", ""
            ),
            "stage22_evidence_notes": stage22_by_id[candidate_id].get(
                "evidence_notes", ""
            ),
            "stage22_correction_notes": stage22_by_id[candidate_id].get(
                "correction_notes", ""
            ),
            "stage22_unresolved_issues": stage22_by_id[candidate_id].get(
                "unresolved_issues", ""
            ),
        })
        records.append(record)

    ids = [record["candidate_id"] for record in records]
    if len(records) != 21:
        raise ValueError(f"Expected 21 dashboard records; found {len(records)}.")
    if len(ids) != len(set(ids)):
        raise ValueError("Dashboard candidate IDs must be unique.")
    return records


def build_json_export(
    records: Sequence[Mapping[str, Any]],
    draft_decisions: Mapping[str, str],
    saved_decisions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a complete export, with draft and saved choices kept separate."""
    saved = saved_decisions or {}
    exported = []
    for record in records:
        candidate_id = str(record["candidate_id"])
        draft = draft_decisions.get(candidate_id, "PENDING")
        if draft not in DECISIONS:
            raise ValueError(f"Invalid draft decision for {candidate_id}: {draft}")
        saved_decision = saved.get(candidate_id)
        if saved_decision is not None and saved_decision not in DECISIONS:
            raise ValueError(
                f"Invalid saved decision for {candidate_id}: {saved_decision}"
            )
        entry = copy.deepcopy(dict(record))
        entry["review_state"] = {
            "draft_selection": draft,
            "draft_is_saved": saved_decision == draft,
            "previously_saved_selection": saved_decision,
            "final_dataset_eligible": False,
        }
        exported.append(entry)
    return {
        "schema_version": "1.0",
        "exported_at_utc": _utc_now(),
        "export_type": "COMPLETE_CANDIDATE_REVIEW_EXPORT",
        "decision_notice": (
            "Dropdown choices are selections only; they do not establish legal "
            "correctness or dataset eligibility."
        ),
        "legal_correctness_notice": EXPORT_NOTICE,
        "candidate_count": len(exported),
        "records": exported,
    }


def build_csv_export(export_data: Mapping[str, Any]) -> str:
    """Create a full-detail CSV; nested evidence fields are JSON encoded."""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in export_data["records"]:
        state = record["review_state"]
        writer.writerow({
            "candidate_id": record["candidate_id"],
            "draft_decision": state["draft_selection"],
            "saved_decision": state["previously_saved_selection"] or "",
            "decision_record_status": (
                "USER_SAVED_SELECTION"
                if state["draft_is_saved"]
                else "DRAFT_NOT_SAVED"
            ),
            "final_dataset_eligible": False,
            "original_question": record["original_question"],
            "original_answer": record["original_answer"],
            "suggested_question": record["proposed_question"],
            "suggested_answer": record["proposed_answer"],
            "original_reference": record["original_reference"],
            "proposed_reference": record["proposed_reference"],
            "source_pdf": record["source_document"],
            "pdf_page_1_based": record["pdf_page_1_based"],
            "supporting_passage": record["supporting_passage"],
            "source_page_context_json": _json_cell(record["source_page_context"]),
            "stage23_audit_concerns": record["quality_or_scope_flags"],
            "duplicate_flags": record["duplicate_flags"],
            "stage19_ai_recommendation": record["ai_recommendation"],
            "stage16_original_record_json": _json_cell(
                record["stage16_original_record"]
            ),
            "stage22_corrected_record_json": _json_cell(
                record["stage22_corrected_record"]
            ),
            "stage23_audit_record_json": _json_cell(
                record["stage23_audit_record"]
            ),
        })
    return output.getvalue()


def build_external_review_report(
    records: Sequence[Mapping[str, Any]],
    draft_decisions: Mapping[str, str],
) -> str:
    """Build readable, untruncated text for external AI-assisted review."""
    sections = [
        "# Legal QA batch review request",
        "",
        f"Candidates: {len(records)}",
        "",
        f"**Caution:** {EXPORT_NOTICE}",
        "",
        "The recommendation field is an AI screening suggestion only. "
        "The user decision is shown separately and may still be a draft.",
    ]
    for index, record in enumerate(records, start=1):
        candidate_id = str(record["candidate_id"])
        sections.extend([
            "",
            "---",
            "",
            f"## {index}. Candidate `{candidate_id}`",
            "",
            f"- Source: `{record['source_document']}`, PDF page "
            f"{record['pdf_page_1_based']}",
            f"- Original reference: {record['original_reference'] or '[missing]'}",
            f"- Proposed reference: {record['proposed_reference'] or '[missing]'}",
            f"- Reference status/flags: {record['reference_status']}; "
            f"{record['reference_confidence_flags']}",
            f"- Evidence traceability status: {record['evidence_status']}",
            f"- Stage 22 reference check: "
            f"{record['stage22_reference_check_status']}",
            f"- AI recommendation (not a legal conclusion): "
            f"{record['ai_recommendation']}",
            f"- User decision draft (not submitted): "
            f"{draft_decisions.get(candidate_id, 'PENDING')}",
            f"- Duplicate flags: {record['duplicate_flags'] or '[none]'}",
            f"- Stage 23 concerns: "
            f"{record['quality_or_scope_flags'] or '[none recorded]'}",
            f"- Stage 22 evidence notes: "
            f"{record['stage22_evidence_notes'] or '[none recorded]'}",
            f"- Stage 22 correction notes: "
            f"{record['stage22_correction_notes'] or '[none recorded]'}",
            f"- Stage 22 unresolved issues: "
            f"{record['stage22_unresolved_issues'] or '[none recorded]'}",
            "",
            "### Original question",
            record["original_question"],
            "",
            "### Original answer",
            record["original_answer"],
            "",
            "### Suggested question",
            record["proposed_question"],
            "",
            "### Suggested answer",
            record["proposed_answer"],
            "",
            "### Stage 16 supporting passage",
            record["supporting_passage"],
            "",
            "### Stage 23 source excerpt",
            record["source_provision_excerpt"],
        ])
        if record["adjacent_page_excerpt_if_needed"]:
            sections.extend([
                "",
                "### Stage 23 adjacent-page excerpt",
                record["adjacent_page_excerpt_if_needed"],
            ])
        sections.extend([
            "",
            "### Complete extracted source page context",
        ])
        for page in record["source_page_context"]:
            sections.extend([
                "",
                f"#### PDF page {page['pdf_page_1_based']}",
                page["text"],
            ])
        sections.extend([
            "",
            "### Full Stage 23 audit record",
            "```json",
            json.dumps(
                record["stage23_audit_record"],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "### Full Stage 22 corrected-review record",
            "```json",
            json.dumps(
                record["stage22_corrected_record"],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
        ])
    return "\n".join(sections).rstrip() + "\n"


def build_saved_decisions(
    records: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, str],
) -> dict[str, Any]:
    """Represent explicitly saved dropdown choices, never as dataset approvals."""
    missing = {str(record["candidate_id"]) for record in records} - set(decisions)
    if missing:
        raise ValueError(f"Missing saved decisions for candidate IDs: {sorted(missing)}")
    entries = []
    for record in records:
        candidate_id = str(record["candidate_id"])
        decision = decisions[candidate_id]
        if decision not in DECISIONS:
            raise ValueError(f"Invalid saved decision for {candidate_id}: {decision}")
        entries.append({
            "candidate_id": candidate_id,
            "decision": decision,
            "candidate_record": copy.deepcopy(dict(record)),
            "decision_status": "USER_SAVED_SELECTION_NOT_APPROVAL",
            "final_dataset_eligible": False,
        })
    return {
        "schema_version": "1.0",
        "saved_at_utc": _utc_now(),
        "save_type": "EXPLICIT_USER_SAVED_DROPDOWN_SELECTIONS",
        "legal_correctness_notice": EXPORT_NOTICE,
        "candidate_count": len(entries),
        "decisions": entries,
    }


def write_saved_decisions(
    output_root: Path,
    payload: Mapping[str, Any],
    full_export: Mapping[str, Any] | None = None,
) -> Path:
    """Persist one immutable timestamped save snapshot."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_root / f"legal_qa_stage25_dashboard_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "saved_review_selections.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if full_export is not None:
        export_path = output_dir / "complete_review_export.json"
        with export_path.open("x", encoding="utf-8") as stream:
            json.dump(full_export, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return path


def write_exports(
    output_root: Path,
    json_text: str,
    csv_text: str,
) -> tuple[Path, Path]:
    """Write a non-overwriting JSON/CSV export pair to a timestamped directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_root / f"legal_qa_stage25_export_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "complete_review_export.json"
    csv_path = output_dir / "complete_review_export.csv"
    with json_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json_text)
        stream.write("\n")
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        stream.write(csv_text)
    return json_path, csv_path
