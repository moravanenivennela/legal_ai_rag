"""Prepare and import human review decisions for source-grounded QA candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_source_grounded_qa_candidates import (
    HUMAN_REVIEW_STATUS,
    LEGAL_REFERENCE_RE,
    NEAR_DUPLICATE_THRESHOLD,
    ROOT,
    SOURCE_PDFS,
    normalize_text,
    reference_supported,
    validate_candidate,
)


AUTHORITATIVE_RUN = ROOT / "reports" / "qa_candidate_generation_stage13_20261004T091039Z_seed42"
DECISIONS = frozenset({
    "APPROVE",
    "EDIT_AND_RECHECK",
    "REJECT",
    "NEEDS_MORE_EVIDENCE",
})
REVIEWED_STATUS = "REVIEWED_AND_DETERMINISTICALLY_VALIDATED"
PENDING_STATUS = "PENDING_HUMAN_REVIEW"
EDITED_PENDING_STATUS = "EDITED_PENDING_REAPPROVAL"
REJECTED_STATUS = "REJECTED"
INVALID_STATUS = "REJECTED_DETERMINISTIC_VALIDATION"
REVIEW_COLUMNS = [
    "candidate_id",
    "question",
    "answer",
    "reviewed_answer",
    "evidence",
    "source_filename",
    "page",
    "legal_reference",
    "reference_status",
    "reference_confidence_flags",
    "duplicate_flags",
    "duplicate_group_id",
    "duplicate_candidate_ids",
    "source_reference_conflict",
    "source_reference_conflict_group_id",
    "source_reference_conflict_candidate_ids",
    "source_reference_conflict_reason",
    "question_type",
    "review_decision",
    "reviewer",
    "reason",
    "reviewer_notes",
]
REQUIRED_DECISION_FIELDS = ("reviewer", "reason")


def _json_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _normalized_reference(candidate: Mapping[str, Any]) -> str:
    return normalize_text(candidate.get("legal_reference"))


def _reference_confidence_flags(candidate: Mapping[str, Any]) -> list[str]:
    flags = []
    status = candidate.get("reference_status")
    if status == "EXPLICIT_NUMBERED_HEADING":
        flags.append("EXPLICIT_HEADING_MATCH_NOT_LEGAL_APPROVAL")
    elif status == "UNCONFIRMED":
        flags.append("REFERENCE_UNCONFIRMED")
    else:
        flags.append("UNRECOGNIZED_REFERENCE_STATUS")
    reference = candidate.get("legal_reference")
    if reference and not LEGAL_REFERENCE_RE.fullmatch(str(reference)):
        flags.append("REFERENCE_FORMAT_CONCERN")
    return flags


def _source_reference_reason(candidate: Mapping[str, Any]) -> str:
    reference = str(candidate.get("legal_reference") or "")
    if not reference:
        return ""
    source = Path(str(candidate.get("source_filename") or "")).name
    if reference.casefold().startswith("article "):
        if source != "constitution_of_india.pdf":
            return "Article reference is paired with a non-Constitution source."
    elif reference.casefold().startswith("section "):
        if source != "consumer_protection_act_2019.pdf":
            return "Section reference is paired with a non-Consumer-Act source."
    else:
        return "Reference prefix is not recognized."
    if not LEGAL_REFERENCE_RE.fullmatch(reference):
        return "Reference does not match the supported Article/Section format."
    if not reference_supported(source, str(candidate.get("evidence") or ""), reference):
        return "Declared reference is not supported by the linked evidence."
    return ""


def group_review_candidates(
    candidates: Sequence[Mapping[str, Any]],
    near_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return worksheet rows with deterministic duplicate and reference flags."""
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be greater than 0 and at most 1.")

    pair_reasons: dict[str, set[str]] = defaultdict(set)
    related_ids: dict[str, set[str]] = defaultdict(set)
    normalized_questions = {
        str(row["candidate_id"]): normalize_text(row.get("question"))
        for row in candidates
    }
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        question = normalized_questions[str(row["candidate_id"])]
        if question:
            exact_groups[question].append(str(row["candidate_id"]))
    for ids in exact_groups.values():
        if len(ids) > 1:
            for candidate_id in ids:
                pair_reasons[candidate_id].add("EXACT_QUESTION_DUPLICATE")
                related_ids[candidate_id].update(
                    other_id for other_id in ids if other_id != candidate_id
                )

    rows_by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        rows_by_type[str(row.get("question_type") or "")].append(row)
    for typed_rows in rows_by_type.values():
        for index, left in enumerate(typed_rows):
            left_id = str(left["candidate_id"])
            left_question = normalized_questions[left_id]
            if not left_question:
                continue
            for right in typed_rows[index + 1:]:
                right_id = str(right["candidate_id"])
                right_question = normalized_questions[right_id]
                if not right_question or left_question == right_question:
                    continue
                if (
                    abs(len(left_question) - len(right_question))
                    / max(len(left_question), len(right_question))
                    > 1 - near_threshold
                ):
                    continue
                if SequenceMatcher(
                    None,
                    left_question,
                    right_question,
                ).ratio() >= near_threshold:
                    pair_reasons[left_id].add("NEAR_QUESTION_DUPLICATE")
                    pair_reasons[right_id].add("NEAR_QUESTION_DUPLICATE")
                    related_ids[left_id].add(right_id)
                    related_ids[right_id].add(left_id)

    by_passage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        passage_id = str(row.get("passage_id") or "")
        if passage_id:
            by_passage[passage_id].append(row)
    conflict_reasons: dict[str, set[str]] = defaultdict(set)
    conflict_related_ids: dict[str, set[str]] = defaultdict(set)
    conflict_group_ids: dict[str, str] = {}
    for passage_rows in by_passage.values():
        passage_candidate_ids = {
            str(row["candidate_id"]) for row in passage_rows
        }
        explicit_refs = {
            _normalized_reference(row)
            for row in passage_rows
            if row.get("legal_reference")
        }
        if len(explicit_refs) > 1:
            for row in passage_rows:
                if row.get("legal_reference"):
                    candidate_id = str(row["candidate_id"])
                    conflict_reasons[candidate_id].add(
                        "CONFLICTING_REFERENCES_WITHIN_PASSAGE"
                    )
                    conflict_related_ids[candidate_id].update(
                        passage_candidate_ids - {candidate_id}
                    )
                    conflict_group_ids[candidate_id] = "ref-" + hashlib.sha256(
                        str(row.get("passage_id")).encode("utf-8")
                    ).hexdigest()[:12]
        for row in passage_rows:
            candidate_id = str(row["candidate_id"])
            reason = _source_reference_reason(row)
            if reason:
                conflict_reasons[candidate_id].add(reason)
                conflict_related_ids[candidate_id].update(
                    passage_candidate_ids - {candidate_id}
                )
                conflict_group_ids.setdefault(
                    candidate_id,
                    "ref-" + hashlib.sha256(
                        str(row.get("passage_id")).encode("utf-8")
                    ).hexdigest()[:12],
                )

    candidate_group: dict[str, set[str]] = {}
    visited = set()
    for candidate_id in related_ids:
        if candidate_id in visited:
            continue
        stack = [candidate_id]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(related_ids[current] - component)
        visited.update(component)
        if len(component) > 1:
            group_id = "dup-" + hashlib.sha256(
                "\0".join(sorted(component)).encode("utf-8")
            ).hexdigest()[:12]
            for member_id in component:
                candidate_group[member_id] = {group_id}
    results = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        question = normalized_questions[candidate_id]
        duplicate_ids = sorted(related_ids[candidate_id])
        results.append({
            "candidate_id": candidate_id,
            "question": str(candidate.get("question") or ""),
            "answer": str(candidate.get("answer") or ""),
            "reviewed_answer": "",
            "evidence": str(candidate.get("evidence") or ""),
            "source_filename": str(candidate.get("source_filename") or ""),
            "page": candidate.get("page", ""),
            "legal_reference": candidate.get("legal_reference") or "",
            "reference_status": candidate.get("reference_status") or "",
            "reference_confidence_flags": "; ".join(
                _reference_confidence_flags(candidate)
            ),
            "duplicate_flags": "; ".join(sorted(
                set(candidate.get("duplicate_flags") or [])
                | pair_reasons[candidate_id]
            )),
            "duplicate_group_id": next(iter(candidate_group.get(candidate_id, set())), ""),
            "duplicate_candidate_ids": "; ".join(duplicate_ids),
            "source_reference_conflict": bool(
                conflict_reasons[candidate_id]
            ),
            "source_reference_conflict_group_id": conflict_group_ids.get(
                candidate_id,
                "",
            ),
            "source_reference_conflict_candidate_ids": "; ".join(
                sorted(conflict_related_ids[candidate_id])
            ),
            "source_reference_conflict_reason": "; ".join(sorted(
                conflict_reasons[candidate_id]
            )),
            "question_type": str(candidate.get("question_type") or ""),
            "review_decision": "",
            "reviewer": "",
            "reason": "",
            "reviewer_notes": "",
        })
    return results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object.")
            result.append(value)
    return result


def _write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def prepare_review_workspace(
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Copy selected records to a blank worksheet without changing source files."""
    manifest_path = source_dir / "run_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("counts", {}).get("retained_candidates") != 1000:
        raise ValueError("Authoritative input must declare exactly 1,000 selected candidates.")
    source_path = source_dir / "candidates.jsonl"
    candidates = _read_jsonl(source_path)
    if len(candidates) != 1000:
        raise ValueError(f"Expected 1,000 selected rows; found {len(candidates)}.")
    seen_ids = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_ids:
            raise ValueError("Candidate IDs must be present and unique.")
        seen_ids.add(candidate_id)
        if candidate.get("review_status") != HUMAN_REVIEW_STATUS:
            raise ValueError(
                f"Candidate {candidate_id} is not pending human review."
            )

    worksheet_rows = group_review_candidates(candidates)
    output_dir.mkdir(parents=True, exist_ok=False)
    worksheet_path = output_dir / "review_worksheet.csv"
    _write_csv_exclusive(worksheet_path, worksheet_rows)
    guide_path = output_dir / "review_instructions.md"
    with guide_path.open("x", encoding="utf-8") as stream:
        stream.write(
            "# Legal QA candidate review\n\n"
            "Review each question and answer against the complete evidence "
            "passage and the cited source page. Automated page, reference, "
            "schema, and duplicate flags are review aids only; they do not "
            "establish legal correctness.\n\n"
            "Use `APPROVE`, `EDIT_AND_RECHECK`, `REJECT`, or "
            "`NEEDS_MORE_EVIDENCE` in `review_decision`. Enter a reviewer "
            "identifier and a concise reason for every completed decision. "
            "Put a proposed revision in `reviewed_answer`. An edited answer "
            "stays out of the reviewed dataset unless it passes deterministic "
            "source/page/reference/evidence checks and receives an explicit "
            "approval decision. Duplicate and reference-conflict groups are "
            "identified by their group IDs; inspect related candidate IDs "
            "together.\n\n"
            "Leave undecided records blank. Do not treat an empty decision, "
            "page match, heading match, or this worksheet as approval.\n\n"
            "After saving the completed CSV, import it with:\n\n"
            "```powershell\n"
            "& .\\venv\\Scripts\\python.exe "
            "scripts\\legal_qa_human_review.py import "
            "--review-csv <path-to-review_worksheet.csv> "
            "--output-dir <new-empty-output-directory>\n"
            "```\n\n"
            "The import command creates a separate reviewed dataset containing "
            "only explicit approvals that pass deterministic validation. "
            "Rejected, edited-pending, uncertain, invalid, and undecided rows "
            "remain excluded.\n"
        )
    summary = {
        "source_run": _display_path(source_dir),
        "selected_count": len(candidates),
        "reviewed_count": 0,
        "approved_count": 0,
        "rejected_count": 0,
        "edited_count": 0,
        "still_pending_count": len(candidates),
        "eligible_reviewed_dataset_count": 0,
        "legal_correctness_claim": (
            "None. The worksheet and automated checks do not establish legal "
            "correctness. Approval requires an explicit reviewer decision."
        ),
        "outputs": {
            "worksheet": worksheet_path.name,
            "instructions": guide_path.name,
        },
    }
    _write_json_exclusive(output_dir / "review_summary.json", summary)
    workspace_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": _display_path(source_dir),
        "source_manifest": _display_path(source_dir / "run_manifest.json"),
        "candidate_count": len(candidates),
        "decision_options": sorted(DECISIONS),
        "review_policy": {
            "no_automatic_approval": True,
            "approval_requires_reviewer_and_reason": True,
            "edits_require_deterministic_revalidation_and_reapproval": True,
            "only_approved_and_validated_records_enter_reviewed_dataset": True,
        },
        "outputs": {
            "worksheet": worksheet_path.name,
            "instructions": guide_path.name,
            "summary": "review_summary.json",
        },
    }
    _write_json_exclusive(output_dir / "review_workspace_manifest.json", workspace_manifest)
    return summary


def _load_source_pages_for_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], str]:
    needed: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        try:
            page = int(row.get("page", ""))
        except (TypeError, ValueError):
            continue
        needed[str(row.get("source_filename") or "")].add(page)

    source_texts = {}
    for source, page_numbers in needed.items():
        pdf_path = SOURCE_PDFS.get(source)
        if pdf_path is None:
            continue
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Source PDF is missing: {pdf_path}")
        with pymupdf.open(pdf_path) as document:
            for page_number in page_numbers:
                if 1 <= page_number <= len(document):
                    source_texts[(source, page_number)] = document[
                        page_number - 1
                    ].get_text("text", sort=True)
    return source_texts


def _validate_review_decision(row: Mapping[str, Any]) -> list[str]:
    decision = str(row.get("review_decision") or "").strip().upper()
    if not decision:
        return []
    errors = []
    if decision not in DECISIONS:
        return [f"Unsupported review decision: {decision}."]
    for field in REQUIRED_DECISION_FIELDS:
        if not str(row.get(field) or "").strip():
            errors.append(f"{field} is required for a completed decision.")
    if decision == "APPROVE" and str(row.get("reviewed_answer") or "").strip():
        if not normalize_text(row["reviewed_answer"]):
            errors.append("APPROVE cannot contain an empty reviewed_answer.")
    if decision == "EDIT_AND_RECHECK" and not str(
        row.get("reviewed_answer") or ""
    ).strip():
        errors.append("EDIT_AND_RECHECK requires a non-empty reviewed_answer.")
    return errors


def validate_reviewed_answer(
    candidate: Mapping[str, Any],
    answer: str,
    source_page_texts: Mapping[tuple[str, int], str],
) -> list[str]:
    """Check the candidate's source/page, supported reference, and answer evidence."""
    updated = dict(candidate)
    updated["answer"] = answer
    updated["review_status"] = HUMAN_REVIEW_STATUS
    return validate_candidate(updated, source_page_texts)


def import_review_decisions(
    source_dir: Path,
    review_csv: Path,
    output_dir: Path,
    source_page_texts: Mapping[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    """Validate decisions, keeping pending/edited/rejected rows out of approved data."""
    source_candidates = _read_jsonl(source_dir / "candidates.jsonl")
    expected = {str(row["candidate_id"]): row for row in source_candidates}
    if len(expected) != len(source_candidates):
        raise ValueError("Source candidate IDs must be unique.")
    for candidate_id, candidate in expected.items():
        if candidate.get("review_status") != HUMAN_REVIEW_STATUS:
            raise ValueError(
                f"Source candidate {candidate_id} is not pending human review."
            )
    with review_csv.open(encoding="utf-8-sig", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    by_id: dict[str, dict[str, Any]] = {}
    for row in review_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id not in expected:
            raise ValueError(f"Unknown or empty candidate_id: {candidate_id!r}.")
        if candidate_id in by_id:
            raise ValueError(f"Duplicate review row for candidate {candidate_id}.")
        by_id[candidate_id] = row

    if source_page_texts is None:
        source_page_texts = _load_source_pages_for_rows(review_rows)
    reviewed_dataset = []
    imported_rows = []
    errors = []
    counts: Counter[str] = Counter()
    for candidate_id, candidate in expected.items():
        review = by_id.get(candidate_id)
        if review is None:
            counts["still_pending"] += 1
            continue
        decision_errors = _validate_review_decision(review)
        if decision_errors:
            errors.extend(f"{candidate_id}: {error}" for error in decision_errors)
            counts["decision_errors"] += 1
            continue
        decision = str(review.get("review_decision") or "").strip().upper()
        if not decision:
            counts["still_pending"] += 1
            continue

        counts["reviewed"] += 1
        imported = dict(candidate)
        imported["review_decision"] = decision
        imported["reviewer"] = str(review.get("reviewer") or "").strip()
        imported["review_reason"] = str(review.get("reason") or "").strip()
        imported["reviewer_notes"] = str(review.get("reviewer_notes") or "").strip()
        imported["reviewed_answer"] = str(
            review.get("reviewed_answer") or ""
        ).strip()
        if decision == "APPROVE":
            approved_answer = str(
                review.get("reviewed_answer") or candidate.get("answer") or ""
            ).strip()
            validation_errors = validate_reviewed_answer(
                candidate,
                approved_answer,
                source_page_texts,
            )
            if validation_errors:
                imported["review_status"] = INVALID_STATUS
                imported["deterministic_validation_errors"] = validation_errors
                counts["validation_rejected"] += 1
                errors.extend(
                    f"{candidate_id}: {error}" for error in validation_errors
                )
            else:
                imported["answer"] = approved_answer
                imported["review_status"] = REVIEWED_STATUS
                imported["deterministic_validation_errors"] = []
                reviewed_dataset.append(imported)
                counts["approved_and_validated"] += 1
        elif decision == "EDIT_AND_RECHECK":
            edited_answer = str(review.get("reviewed_answer") or "").strip()
            imported["answer"] = edited_answer
            validation_errors = validate_reviewed_answer(
                candidate,
                edited_answer,
                source_page_texts,
            )
            imported["deterministic_validation_errors"] = validation_errors
            if validation_errors:
                imported["review_status"] = INVALID_STATUS
                counts["validation_rejected"] += 1
                errors.extend(
                    f"{candidate_id}: {error}" for error in validation_errors
                )
            else:
                imported["review_status"] = EDITED_PENDING_STATUS
                counts["edited_pending_reapproval"] += 1
        elif decision == "REJECT":
            imported["review_status"] = REJECTED_STATUS
            counts["rejected"] += 1
        else:
            imported["review_status"] = PENDING_STATUS
            counts["needs_more_evidence"] += 1
        imported_rows.append(imported)

    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / "reviewed_qa.jsonl"
    with dataset_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in reviewed_dataset:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit_path = output_dir / "review_import_results.jsonl"
    with audit_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in imported_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_run": _display_path(source_dir),
        "input_candidate_count": len(expected),
        "decision_rows": len(by_id),
        "reviewed_count": counts["reviewed"],
        "approved_count": counts["approved_and_validated"],
        "rejected_count": counts["rejected"] + counts["validation_rejected"],
        "edited_count": counts["edited_pending_reapproval"],
        "needs_more_evidence_count": counts["needs_more_evidence"],
        "still_pending_count": counts["still_pending"],
        "decision_error_count": counts["decision_errors"],
        "validation_rejected_count": counts["validation_rejected"],
        "eligible_reviewed_dataset_count": len(reviewed_dataset),
        "errors": errors,
        "legal_correctness_claim": (
            "No legal correctness is inferred from automation. Included records "
            "have an explicit human APPROVE decision and passed deterministic "
            "schema, page-provenance, reference-format/support, and answer-evidence "
            "checks; legal correctness remains a human responsibility."
        ),
        "outputs": {
            "reviewed_dataset": dataset_path.name,
            "review_import_results": audit_path.name,
        },
    }
    _write_json_exclusive(output_dir / "review_summary.json", summary)
    return summary


def _timestamped_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "reports" / f"{prefix}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-dir", type=Path, default=AUTHORITATIVE_RUN)
    prepare.add_argument("--output-dir", type=Path)
    importer = subparsers.add_parser("import")
    importer.add_argument("--source-dir", type=Path, default=AUTHORITATIVE_RUN)
    importer.add_argument("--review-csv", type=Path, required=True)
    importer.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or _timestamped_dir(
        "legal_qa_review_workspace"
        if args.command == "prepare"
        else "legal_qa_review_import"
    )
    if args.command == "prepare":
        summary = prepare_review_workspace(args.source_dir, output_dir)
    else:
        summary = import_review_decisions(
            args.source_dir,
            args.review_csv,
            output_dir,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
