"""Guide a human through Stage 23 candidates without changing earlier records."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.review_first_batch_legal_qa import (  # noqa: E402
    DECISIONS,
    validate_decision_form,
)
from scripts.interactive_legal_qa_review import (  # noqa: E402
    _read_multiline,
    _write_json_exclusive,
)

STAGE23_DIR = ROOT / "reports" / "legal_qa_stage23_audit_20261004T123528Z"
DEFAULT_AUDIT_CSV = STAGE23_DIR / "ready_batch_audit.csv"
DEFAULT_STAGE22_CSV = (
    ROOT
    / "reports"
    / "legal_qa_stage22_corrected_review_20261004T122942Z"
    / "corrected_review_batch.csv"
)
DEFAULT_FORMS_DIR = (
    ROOT
    / "reports"
    / "legal_qa_individual_review_20261004T094008Z"
    / "candidate_forms"
)
ALLOWED_SOURCES = frozenset({
    "constitution_of_india.pdf",
    "consumer_protection_act_2019.pdf",
})
DECISION_VALUES = frozenset(DECISIONS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_form(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Review form must be a JSON object: {path}")
    return value


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamped_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "reports" / f"legal_qa_stage23_review_{stamp}"


def validate_source_mapping(
    source_document: str,
    page_number: int,
    page_counts: Mapping[str, int],
) -> None:
    if source_document not in ALLOWED_SOURCES:
        raise ValueError(f"Unexpected source document: {source_document}")
    page_count = page_counts.get(source_document)
    if page_count is None or not 1 <= page_number <= page_count:
        raise ValueError(
            f"Invalid PDF page {page_number} for {source_document}."
        )


def validate_guided_decision(
    decision_record: Mapping[str, Any],
    expected_candidate_id: str,
) -> list[str]:
    review = decision_record.get("review")
    candidate = decision_record.get("candidate")
    if not isinstance(review, Mapping) or not isinstance(candidate, Mapping):
        return ["Decision record must contain candidate and review objects."]
    decision = str(review.get("decision") or "").strip().upper()
    if decision not in DECISION_VALUES:
        return ["Decision must be APPROVE, REJECT, EDIT_AND_RECHECK, or PENDING."]
    errors = validate_decision_form(
        {
            "candidate": {
                "candidate_id": candidate.get("candidate_id"),
                "legal_reference": candidate.get("proposed_reference"),
                "reference_status": "EXPLICIT_NUMBERED_HEADING",
                "proposed_answer": candidate.get("proposed_answer"),
            },
            "review": {
                **dict(review),
                "answer_supported_by_passage_confirmation": review.get(
                    "answer_supported_by_complete_provision_confirmation", "NO"
                ),
                "legal_reference_checked_against_original_source": review.get(
                    "legal_reference_inspected_confirmation", "NO"
                ),
                "edited_answer_rechecked_confirmation": review.get(
                    "edited_answer_rechecked_confirmation", "NO"
                ),
            },
        },
        {"candidate_id": expected_candidate_id},
    )
    if str(candidate.get("candidate_id") or "") != expected_candidate_id:
        errors.append("Candidate ID does not match the Stage 23 review batch.")
    if decision == "APPROVE":
        for field, label in (
            ("source_provision_inspected_confirmation", "source provision"),
            ("legal_reference_inspected_confirmation", "legal reference"),
            (
                "answer_supported_by_complete_provision_confirmation",
                "answer support in the complete provision",
            ),
        ):
            if str(review.get(field) or "").strip().upper() != "YES":
                errors.append(f"APPROVE requires YES confirmation for {label}.")
        if not str(review.get("reviewed_reference") or "").strip():
            errors.append(
                "APPROVE requires the exact reference the reviewer checked."
            )
        if not str(review.get("reviewed_question") or "").strip():
            errors.append("APPROVE requires the question wording being approved.")
        if not str(review.get("reviewed_answer") or "").strip():
            errors.append("APPROVE requires the answer wording being approved.")
        if review.get("reviewed_text_source") not in {
            "ORIGINAL",
            "SUGGESTED",
            "REVIEWER_EDIT",
        }:
            errors.append(
                "APPROVE requires identifying the exact wording being reviewed."
            )
        if (
            review.get("reviewed_text_source") == "REVIEWER_EDIT"
            and str(review.get("edited_answer_rechecked_confirmation") or "").upper()
            != "YES"
        ):
            errors.append(
                "A reviewer edit requires explicit YES confirmation of an "
                "independent source recheck."
            )
    if decision == "EDIT_AND_RECHECK" and str(
        review.get("dataset_eligibility") or ""
    ) != "INELIGIBLE_PENDING_INDEPENDENT_RECHECK_AND_APPROVAL":
        errors.append("Edited records must remain ineligible pending recheck.")
    return errors


def load_review_candidates(
    audit_csv: Path,
    stage22_csv: Path,
    forms_dir: Path,
    page_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    audit_rows = read_csv(audit_csv)
    stage22_rows = read_csv(stage22_csv)
    ready_rows = [
        row for row in stage22_rows
        if row.get("priority_group") == "READY_FOR_HUMAN_REVIEW"
    ]
    for label, rows in (("Stage 23 audit", audit_rows), ("Stage 22 ready batch", ready_rows)):
        ids = [str(row.get("candidate_id") or "") for row in rows]
        if any(not candidate_id for candidate_id in ids):
            raise ValueError(f"{label} contains an empty candidate ID.")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label} contains duplicate candidate IDs.")
    if len(audit_rows) != 21 or len(ready_rows) != 21:
        raise ValueError(
            f"Expected 21 candidates in each input; found {len(audit_rows)} "
            f"Stage 23 and {len(ready_rows)} Stage 22."
        )

    stage22_by_id = {row["candidate_id"]: row for row in ready_rows}
    if {row["candidate_id"] for row in audit_rows} != set(stage22_by_id):
        raise ValueError("Stage 23 IDs do not exactly match Stage 22 ready IDs.")

    records = []
    for audit in audit_rows:
        candidate_id = audit["candidate_id"]
        corrected = stage22_by_id[candidate_id]
        form_path = forms_dir / f"{candidate_id}.json"
        form = _read_form(form_path)
        original = form.get("candidate")
        review = form.get("review")
        if not isinstance(original, Mapping) or not isinstance(review, Mapping):
            raise ValueError(f"Stage 16 form is malformed: {form_path}")
        if str(original.get("candidate_id") or "") != candidate_id:
            raise ValueError(f"Stage 16 candidate ID mismatch: {candidate_id}")
        if str(review.get("decision") or "").upper() != "PENDING":
            raise ValueError(f"Stage 16 form is not pending: {candidate_id}")
        if _normalized(original.get("question")) != _normalized(
            audit.get("original_question")
        ) or _normalized(original.get("proposed_answer")) != _normalized(
            audit.get("original_answer")
        ):
            raise ValueError(f"Stage 16 original text mismatch: {candidate_id}")
        for field in ("original_question", "original_answer", "proposed_question", "proposed_answer"):
            if _normalized(audit.get(field)) != _normalized(corrected.get(field)):
                raise ValueError(
                    f"Stage 22/23 {field} mismatch for candidate {candidate_id}."
                )
        source_document = str(audit.get("source_document") or "")
        page_number = int(audit.get("pdf_page_1_based") or 0)
        validate_source_mapping(source_document, page_number, page_counts)
        if (
            source_document != corrected.get("source_document")
            or str(page_number) != str(corrected.get("pdf_page_1_based"))
            or source_document != original.get("source_document")
            or str(page_number) != str(original.get("pdf_page"))
        ):
            raise ValueError(f"Source/page mapping mismatch: {candidate_id}")
        records.append({
            "candidate_id": candidate_id,
            "original_question": audit["original_question"],
            "original_answer": audit["original_answer"],
            "proposed_question": corrected["proposed_question"],
            "proposed_answer": corrected["proposed_answer"],
            "original_reference": audit["original_reference"],
            "proposed_reference": corrected["proposed_reference"],
            "source_document": source_document,
            "pdf_page_1_based": page_number,
            "source_text_audit": audit["source_text_audit"],
            "source_text_finding": audit["source_text_finding"],
            "reference_page_consistency": audit["reference_page_consistency"],
            "quality_or_scope_flags": audit["quality_or_scope_flags"],
            "duplicate_flags": audit["duplicate_flags"],
            "ai_recommendation": audit["first_pass_recommendation"],
            "source_provision_excerpt": audit["source_provision_excerpt"],
            "adjacent_page_excerpt_if_needed": audit[
                "adjacent_page_excerpt_if_needed"
            ],
        })
    return records


def read_pdf_context(
    pdf_path: Path,
    cited_page: int,
) -> list[tuple[int, str]]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Source PDF not found: {pdf_path}")
    with pymupdf.open(pdf_path) as document:
        if not 1 <= cited_page <= document.page_count:
            raise ValueError(
                f"PDF page {cited_page} is outside {pdf_path.name}."
            )
        start = max(1, cited_page - 1)
        end = min(document.page_count, cited_page + 1)
        return [
            (page_number, document[page_number - 1].get_text())
            for page_number in range(start, end + 1)
        ]


def render_candidate(
    record: Mapping[str, Any],
    position: int,
    total: int,
    pdf_path: Path,
    page_context: Sequence[tuple[int, str]],
) -> str:
    question_changed = _normalized(record["original_question"]) != _normalized(
        record["proposed_question"]
    )
    answer_changed = _normalized(record["original_answer"]) != _normalized(
        record["proposed_answer"]
    )
    lines = [
        "",
        "=" * 88,
        f"Candidate {position} of {total} | ID: {record['candidate_id']}",
        "=" * 88,
        "ORIGINAL QUESTION:",
        record["original_question"],
        "",
        "ORIGINAL ANSWER:",
        record["original_answer"],
    ]
    if question_changed or answer_changed:
        lines.extend([
            "",
            "SUGGESTED WORDING (AI-prepared; not approved):",
            f"Question: {record['proposed_question']}",
            f"Answer: {record['proposed_answer']}",
        ])
    lines.extend([
        "",
        f"Source PDF: {pdf_path}",
        f"PDF page (1-based): {record['pdf_page_1_based']}",
        f"Original reference: {record['original_reference'] or '[missing]'}",
        f"Proposed reference: {record['proposed_reference'] or '[missing]'}",
        "AI recommendation (Stage 19 suggestion only; not a legal conclusion): "
        f"{record['ai_recommendation']}",
        f"Stage 23 source-text audit: {record['source_text_audit']}",
        f"Stage 23 evidence finding: {record['source_text_finding']}",
        f"Reference/page note: {record['reference_page_consistency']}",
        f"Concerns: {record['quality_or_scope_flags'] or '[none recorded]'}",
        f"Duplicate flags: {record['duplicate_flags'] or '[none recorded]'}",
        "",
        "SOURCE PROVISION EXCERPT FROM STAGE 23 (for orientation; verify below):",
        record["source_provision_excerpt"],
    ])
    if record["adjacent_page_excerpt_if_needed"]:
        lines.extend([
            "",
            "ADJACENT-PAGE EXCERPT:",
            record["adjacent_page_excerpt_if_needed"],
        ])
    lines.extend([
        "",
        "FULL PDF PAGE CONTEXT (complete extracted pages, not a shortened snippet):",
    ])
    for page_number, text in page_context:
        lines.extend([
            "",
            f"----- PDF PAGE {page_number} -----",
            text or "[No extractable text on this page; inspect the PDF visually.]",
        ])
    lines.extend([
        "",
        "Before APPROVE: inspect the complete provision and nearby qualifications "
        "in the original PDF (including any continuation beyond these pages); "
        "confirm the exact reference and that the answer is supported by the "
        "complete provision. The AI suggestion and page/text matches are not "
        "legal verification.",
        "=" * 88,
    ])
    return "\n".join(lines)


def _ask_yes_no(
    question: str,
    input_fn: Callable[[str], str],
) -> str:
    while True:
        value = input_fn(f"{question} Enter YES or NO: ").strip().upper()
        if value in {"YES", "NO"}:
            return value


def collect_decision(
    record: Mapping[str, Any],
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Collect one explicit human decision; never infer or import AI labels."""
    while True:
        decision = input_fn(
            "Decision (PENDING / APPROVE / REJECT / EDIT_AND_RECHECK): "
        ).strip().upper()
        if decision not in DECISION_VALUES:
            output_fn("Choose one of the four listed decisions.")
            continue

        review: dict[str, Any] = {
            "decision": decision,
            "reviewer": "",
            "reason": "",
            "decision_timestamp_utc": _timestamp(),
            "source_provision_inspected_confirmation": "NO",
            "legal_reference_inspected_confirmation": "NO",
            "answer_supported_by_complete_provision_confirmation": "NO",
            "reviewed_reference": "",
            "reviewed_text_source": "",
            "reviewed_question": "",
            "reviewed_answer": "",
            "edited_answer_rechecked_confirmation": "NO",
            "edited_question": "",
            "edited_answer": "",
            "dataset_eligibility": "INELIGIBLE",
        }
        if decision in {"APPROVE", "REJECT", "EDIT_AND_RECHECK"}:
            review["reviewer"] = input_fn("Reviewer name or identifier: ").strip()
            review["reason"] = input_fn("Reason for this decision: ").strip()
        if decision == "EDIT_AND_RECHECK":
            review["edited_question"] = _read_multiline(
                "Enter the edited question (or keep it unchanged):",
                input_fn,
                output_fn,
            )
            review["edited_answer"] = _read_multiline(
                "Enter the edited answer:",
                input_fn,
                output_fn,
            )
            review["dataset_eligibility"] = (
                "INELIGIBLE_PENDING_INDEPENDENT_RECHECK_AND_APPROVAL"
            )
        elif decision == "APPROVE":
            review["source_provision_inspected_confirmation"] = _ask_yes_no(
                "Did you inspect the complete source provision and its qualifications?",
                input_fn,
            )
            review["legal_reference_inspected_confirmation"] = _ask_yes_no(
                "Did you check the legal reference against the original PDF?",
                input_fn,
            )
            review["answer_supported_by_complete_provision_confirmation"] = _ask_yes_no(
                "Is the answer supported by the complete provision?",
                input_fn,
            )
            review["reviewed_reference"] = input_fn(
                "Enter the exact Article/Section reference you checked: "
            ).strip()
            while review["reviewed_text_source"] not in {
                "ORIGINAL",
                "SUGGESTED",
                "REVIEWER_EDIT",
            }:
                review["reviewed_text_source"] = input_fn(
                    "Which exact wording are you reviewing? "
                    "Enter ORIGINAL, SUGGESTED, or REVIEWER_EDIT: "
                ).strip().upper()
            if review["reviewed_text_source"] == "REVIEWER_EDIT":
                review["edited_answer_rechecked_confirmation"] = _ask_yes_no(
                    "Did you independently recheck the edited answer against "
                    "the complete original provision?",
                    input_fn,
                )
            review["reviewed_question"] = _read_multiline(
                "Enter the exact question wording you are approving:",
                input_fn,
                output_fn,
            )
            review["reviewed_answer"] = _read_multiline(
                "Enter the exact answer wording you are approving:",
                input_fn,
                output_fn,
            )
            if all(
                review[field] == "YES"
                for field in (
                    "source_provision_inspected_confirmation",
                    "legal_reference_inspected_confirmation",
                    "answer_supported_by_complete_provision_confirmation",
                )
            ):
                review["dataset_eligibility"] = (
                    "HUMAN_APPROVAL_RECORDED_NOT_LEGAL_VERIFICATION"
                )

        decision_record = {
            "record_version": "1.0",
            "saved_at_utc": _timestamp(),
            "candidate": copy.deepcopy(dict(record)),
            "review": review,
            "legal_correctness_notice": (
                "Reviewer confirmations are not a guarantee of legal correctness "
                "or current-law status."
            ),
        }
        errors = validate_guided_decision(
            decision_record,
            str(record["candidate_id"]),
        )
        if errors:
            output_fn("This decision cannot be saved as entered:")
            for error in errors:
                output_fn(f"- {error}")
            output_fn("No decision was saved. Please enter a decision again.")
            continue
        return decision_record


def write_instructions(path: Path, output_dir: Path) -> None:
    path.write_text(
        "# Stage 23 guided human review\n\n"
        "Each submitted decision is saved immediately as a separate JSON file "
        "under `decision_records/`; Stage 16 forms and Stage 22/23 reports are "
        "read-only. Use `--start-at N` in a new session to revisit a candidate "
        "without overwriting prior decisions. Press Ctrl+C to stop; decisions "
        "already saved remain in this directory, while the candidate currently "
        "being entered is not saved until submitted.\n\n"
        "`EDIT_AND_RECHECK` is always marked ineligible pending an independent "
        "recheck and a later explicit approval. This workflow does not create "
        "or import a final dataset. Reviewer confirmations do not guarantee "
        "legal correctness or current-law status.\n\n"
        f"Session directory: `{output_dir}`\n",
        encoding="utf-8",
        newline="\n",
    )


def run_review_session(
    audit_csv: Path = DEFAULT_AUDIT_CSV,
    stage22_csv: Path = DEFAULT_STAGE22_CSV,
    forms_dir: Path = DEFAULT_FORMS_DIR,
    output_dir: Path | None = None,
    start_at: int = 1,
    limit: int | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    if start_at < 1:
        raise ValueError("start_at must be at least 1.")
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative.")
    source_paths = {name: ROOT / "data" / name for name in ALLOWED_SOURCES}
    page_counts = {}
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Source PDF not found: {path}")
        with pymupdf.open(path) as document:
            page_counts[name] = document.page_count
    records = load_review_candidates(
        audit_csv,
        stage22_csv,
        forms_dir,
        page_counts,
    )
    if start_at > len(records) + 1:
        raise ValueError("start_at is past the end of the 21-candidate batch.")
    end = len(records) if limit is None else min(len(records), start_at - 1 + limit)
    target_dir = output_dir or timestamped_output_dir()
    target_dir.mkdir(parents=True, exist_ok=False)
    decisions_dir = target_dir / "decision_records"
    decisions_dir.mkdir()
    write_instructions(target_dir / "README.md", target_dir)
    saved = 0
    for index in range(start_at - 1, end):
        record = records[index]
        pdf_path = source_paths[record["source_document"]]
        page_context = read_pdf_context(pdf_path, record["pdf_page_1_based"])
        output_fn(render_candidate(record, index + 1, len(records), pdf_path, page_context))
        decision_record = collect_decision(record, input_fn, output_fn)
        filename = (
            f"{record['candidate_id']}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        _write_json_exclusive(decisions_dir / filename, decision_record)
        saved += 1
    return {
        "session_directory": str(target_dir),
        "decision_records_directory": str(decisions_dir),
        "batch_count": len(records),
        "decisions_saved_this_run": saved,
        "final_dataset_examples_added": 0,
        "final_dataset_created": False,
        "legal_correctness_notice": (
            "Reviewer confirmations are not a guarantee of legal correctness "
            "or current-law status."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--stage22-csv", type=Path, default=DEFAULT_STAGE22_CSV)
    parser.add_argument("--forms-dir", type=Path, default=DEFAULT_FORMS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit cannot be negative.")
    summary = run_review_session(
        audit_csv=args.audit_csv,
        stage22_csv=args.stage22_csv,
        forms_dir=args.forms_dir,
        output_dir=args.output_dir,
        start_at=args.start_at,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
