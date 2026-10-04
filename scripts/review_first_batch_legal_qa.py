"""Create and validate individual human-review forms for the Stage 15 batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_source_grounded_qa_candidates import (  # noqa: E402
    LEGAL_REFERENCE_RE,
    SOURCE_PDFS,
    normalize_text,
    reference_supported,
)

STAGE15_DIR = ROOT / "reports" / "legal_qa_review_prioritization_20261004T093327Z"
STAGE14_DIR = ROOT / "reports" / "legal_qa_review_workspace_20261004T092435Z"
DECISIONS = frozenset({
    "APPROVE",
    "REJECT",
    "EDIT_AND_RECHECK",
    "PENDING",
})
CONFIRMATIONS = frozenset({"YES", "NO"})
COMPLETED_DECISIONS = frozenset({"APPROVE", "REJECT", "EDIT_AND_RECHECK"})
FORM_VERSION = "1.0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object.")
            records.append(record)
    return records


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def validate_decision_form(
    form: Mapping[str, Any],
    expected_candidate: Mapping[str, Any],
) -> list[str]:
    """Check human-entered fields without inferring or supplying a decision."""
    errors = []
    candidate = form.get("candidate")
    review = form.get("review")
    if not isinstance(candidate, Mapping) or not isinstance(review, Mapping):
        return ["Form must contain candidate and review objects."]
    if str(candidate.get("candidate_id") or "") != str(
        expected_candidate.get("candidate_id") or ""
    ):
        errors.append("Candidate ID does not match the review batch.")
    decision = str(review.get("decision") or "").strip().upper()
    if decision not in DECISIONS:
        errors.append("Decision must be APPROVE, REJECT, EDIT_AND_RECHECK, or PENDING.")
        return errors
    if decision in COMPLETED_DECISIONS:
        if not str(review.get("reviewer") or "").strip():
            errors.append("Reviewer is required for a completed decision.")
        if not str(review.get("reason") or "").strip():
            errors.append("A reason is required for a completed decision.")
    for field in (
        "answer_supported_by_passage_confirmation",
        "legal_reference_checked_against_original_source",
        "edited_answer_rechecked_confirmation",
    ):
        value = str(review.get(field) or "").strip().upper()
        if value not in CONFIRMATIONS:
            errors.append(f"{field} must be YES or NO.")
    if decision == "APPROVE":
        if (
            not str(candidate.get("legal_reference") or "").strip()
            or candidate.get("reference_status") == "UNCONFIRMED"
        ) and not str(review.get("reviewed_legal_reference") or "").strip():
            errors.append(
                "An unconfirmed reference must remain pending unless the reviewer "
                "records the exact reference checked in the original source."
            )
        if str(review.get("answer_supported_by_passage_confirmation")).upper() != "YES":
            errors.append(
                "APPROVE requires explicit YES confirmation that the answer is "
                "supported by the passage."
            )
        if str(review.get("legal_reference_checked_against_original_source")).upper() != "YES":
            errors.append(
                "APPROVE requires explicit YES confirmation that the legal "
                "reference was checked against the original source."
            )
        edited_answer = str(review.get("edited_answer") or "").strip()
        is_changed_answer = bool(
            edited_answer
            and normalize_text(edited_answer)
            != normalize_text(candidate.get("proposed_answer"))
        )
        if is_changed_answer and (
            str(review.get("edited_answer_rechecked_confirmation") or "").upper()
            != "YES"
        ):
            errors.append(
                "A changed answer requires explicit YES confirmation that the "
                "edited answer was rechecked."
            )
    if decision == "EDIT_AND_RECHECK" and not str(
        review.get("edited_answer") or ""
    ).strip():
        errors.append("EDIT_AND_RECHECK requires a proposed edited_answer.")
    return errors


def _candidate_from_batch_row(
    row: Mapping[str, Any],
    prioritized_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id") or "")
    details = prioritized_by_id.get(candidate_id)
    if details is None:
        raise ValueError(f"Batch candidate {candidate_id!r} is missing from priority report.")
    if row.get("review_status") != "PENDING_HUMAN_REVIEW":
        raise ValueError(f"Candidate {candidate_id} is not pending human review.")
    if row.get("review_decision") or row.get("reviewer") or row.get("reason"):
        raise ValueError(f"Candidate {candidate_id} already contains review decisions.")
    return {
        "candidate_id": candidate_id,
        "question": row.get("question", ""),
        "proposed_answer": row.get("answer", ""),
        "supporting_passage": row.get("evidence", ""),
        "source_document": row.get("source_filename", ""),
        "pdf_page": row.get("page", ""),
        "legal_reference": row.get("legal_reference", ""),
        "reference_status": row.get("reference_status", ""),
        "reference_confidence_flags": row.get("reference_confidence_flags", ""),
        "evidence_status": row.get("evidence_status", ""),
        "question_type": row.get("question_type", ""),
        "duplicate_group_id": row.get("duplicate_group_id", ""),
        "duplicate_candidate_ids": row.get("duplicate_candidate_ids", ""),
        "duplicate_flags": row.get("duplicate_flags", ""),
        "answer_template_repetition": row.get("answer_template_repetition", ""),
        "priority_rank": row.get("priority_rank", ""),
        "priority_reasons": row.get("priority_reasons", ""),
        "legal_correctness_notice": (
            "Not verified. Automated evidence/reference matching and priority "
            "scores are not proof of legal correctness."
        ),
    }


def _candidate_form_mismatches(
    form_candidate: Mapping[str, Any],
    batch_row: Mapping[str, Any],
) -> list[str]:
    expected_fields = {
        "candidate_id": "candidate_id",
        "question": "question",
        "proposed_answer": "answer",
        "supporting_passage": "evidence",
        "source_document": "source_filename",
        "pdf_page": "page",
        "legal_reference": "legal_reference",
        "reference_status": "reference_status",
        "reference_confidence_flags": "reference_confidence_flags",
        "evidence_status": "evidence_status",
        "question_type": "question_type",
        "duplicate_group_id": "duplicate_group_id",
        "duplicate_candidate_ids": "duplicate_candidate_ids",
        "duplicate_flags": "duplicate_flags",
        "answer_template_repetition": "answer_template_repetition",
        "priority_rank": "priority_rank",
        "priority_reasons": "priority_reasons",
    }
    return [
        form_field
        for form_field, batch_field in expected_fields.items()
        if str(form_candidate.get(form_field) or "")
        != str(batch_row.get(batch_field) or "")
    ]


def prepare_individual_review_workspace(
    batch_csv: Path,
    prioritized_csv: Path,
    stage14_workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create one editable JSON review card per pending candidate."""
    stage14_manifest_path = stage14_workspace / "review_workspace_manifest.json"
    stage14_manifest = json.loads(
        stage14_manifest_path.read_text(encoding="utf-8")
    )
    if stage14_manifest.get("candidate_count") != 1000:
        raise ValueError("Stage 14 review workspace must cover 1,000 candidates.")
    stage14_summary = json.loads(
        (stage14_workspace / "review_summary.json").read_text(encoding="utf-8")
    )
    if (
        stage14_summary.get("reviewed_count") != 0
        or stage14_summary.get("approved_count") != 0
    ):
        raise ValueError("Stage 14 workspace is no longer an untouched blank worksheet.")

    batch_rows = read_csv(batch_csv)
    if len(batch_rows) != 50:
        raise ValueError(f"Expected a 50-candidate first batch; found {len(batch_rows)}.")
    priority_rows = read_csv(prioritized_csv)
    priority_by_id = {
        str(row.get("candidate_id") or ""): row for row in priority_rows
    }
    if len(priority_by_id) != len(priority_rows):
        raise ValueError("Prioritized candidate IDs must be unique.")
    ids = [str(row.get("candidate_id") or "") for row in batch_rows]
    if any(not candidate_id for candidate_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("First-batch candidate IDs must be nonempty and unique.")

    prepared = []
    for batch_row in batch_rows:
        candidate = _candidate_from_batch_row(batch_row, priority_by_id)
        candidate_id = candidate["candidate_id"]
        prepared.append({
            "form_version": FORM_VERSION,
            "candidate": candidate,
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
        })

    forms_dir = output_dir / "candidate_forms"
    output_dir.mkdir(parents=True, exist_ok=False)
    forms_dir.mkdir()
    for form in prepared:
        file_path = forms_dir / f"{form['candidate']['candidate_id']}.json"
        _write_json_exclusive(file_path, form)

    instructions = (
        "# Individual legal QA candidate review\n\n"
        "Open one JSON file in `candidate_forms` at a time. Each file shows the "
        "question, proposed answer, complete supporting passage, source/page, "
        "citation metadata, and duplicate/template warnings. Edit only its "
        "`review` object; candidate details identify the immutable Stage 15 "
        "record.\n\n"
        "Set `decision` to `APPROVE`, `REJECT`, `EDIT_AND_RECHECK`, or `PENDING`. "
        "Every completed decision requires a reviewer identifier and a concise "
        "reason. For `APPROVE`, set both confirmation fields to `YES` only after "
        "personally checking that the answer is supported by the passage and "
        "checking the legal reference against the original legal source. "
        "Unconfirmed references and uncertain interpretations remain pending "
        "until that source check is complete. Automated page matches and scores "
        "are not legal verification.\n\n"
        "`EDIT_AND_RECHECK` requires `edited_answer`; it does not approve the "
        "candidate. Submit a later review form with an explicit `APPROVE` only "
        "after reviewing the edit. Do not change candidate fields or silently "
        "correct citations. When later approving a changed answer, set "
        "`edited_answer_rechecked_confirmation` to `YES` only after rechecking "
        "it. A revised citation may be entered as `reviewed_legal_reference` "
        "only after checking the original source; it must also pass the "
        "deterministic evidence check.\n\n"
        "To validate/import completed forms into a separate audit output, run:\n\n"
        "```powershell\n"
        "& .\\venv\\Scripts\\python.exe "
        "scripts\\review_first_batch_legal_qa.py import "
        "--forms-dir <this-output>\\candidate_forms "
        "--batch-csv <Stage-15-first_review_batch.csv> "
        "--output-dir <new-empty-output-directory>\n"
        "```\n\n"
        "Only explicit approvals with both confirmations and passing "
        "deterministic provenance/evidence checks are emitted to the separate "
        "approved-review file. That file is not a train/validation/test split "
        "and does not establish legal correctness by itself.\n"
    )
    (output_dir / "review_instructions.md").open(
        "x", encoding="utf-8"
    ).write(instructions)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_batch": _display_path(batch_csv),
        "source_prioritization_report": _display_path(prioritized_csv),
        "stage14_workspace": _display_path(stage14_workspace),
        "form_count": len(prepared),
        "reviewed_count": 0,
        "approved_count": 0,
        "rejected_count": 0,
        "edited_count": 0,
        "pending_count": len(prepared),
        "all_decisions_pending": True,
        "legal_correctness_claim": (
            "None. Human confirmations are required; automated matching is not "
            "legal verification."
        ),
        "outputs": {
            "forms_directory": "candidate_forms",
            "instructions": "review_instructions.md",
            "summary": "review_workspace_summary.json",
        },
    }
    _write_json_exclusive(output_dir / "review_workspace_summary.json", summary)
    return summary


def _normalized_substring(needle: Any, haystack: Any) -> bool:
    normalized_needle = normalize_text(needle)
    return bool(normalized_needle) and normalized_needle in normalize_text(haystack)


def validate_approved_form(
    form: Mapping[str, Any],
    source_page_texts: Mapping[tuple[str, int], str],
) -> list[str]:
    """Perform deterministic safety checks after explicit human approval."""
    candidate = form["candidate"]
    review = form["review"]
    answer = str(review.get("edited_answer") or candidate["proposed_answer"]).strip()
    reference = str(
        review.get("reviewed_legal_reference")
        or candidate.get("legal_reference")
        or ""
    ).strip()
    errors = []
    try:
        page_number = int(candidate["pdf_page"])
    except (TypeError, ValueError):
        errors.append("PDF page must be a positive integer.")
    else:
        key = (str(candidate["source_document"]), page_number)
        page_text = source_page_texts.get(key)
        if page_text is None:
            errors.append("Source PDF page could not be loaded.")
        elif not _normalized_substring(candidate["supporting_passage"], page_text):
            errors.append("Supporting passage does not match the declared PDF page.")
    if not _normalized_substring(answer, candidate["supporting_passage"]):
        errors.append("Answer is not found in the supporting passage.")
    if reference:
        if not LEGAL_REFERENCE_RE.fullmatch(reference):
            errors.append("Legal reference does not match Article/Section format.")
        elif not reference_supported(
            str(candidate["source_document"]),
            str(candidate["supporting_passage"]),
            reference,
        ):
            errors.append("Legal reference is not supported by the supplied passage.")
    return errors


def process_review_forms(
    forms_dir: Path,
    batch_csv: Path,
    output_dir: Path,
    source_page_texts: Mapping[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    """Validate submitted forms; never infer reviewer decisions or confirmations."""
    batch_rows = read_csv(batch_csv)
    expected = {str(row["candidate_id"]): row for row in batch_rows}
    form_paths = sorted(forms_dir.glob("*.json"))
    forms_by_id = {}
    errors = []
    for path in form_paths:
        try:
            form = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"{path.name}: unable to read valid JSON form ({error}).")
            continue
        candidate = form.get("candidate", {})
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id not in expected:
            errors.append(f"{path.name}: candidate ID is not in the first batch.")
            continue
        if candidate_id in forms_by_id:
            errors.append(f"{path.name}: duplicate form for {candidate_id}.")
            continue
        if path.stem != candidate_id:
            errors.append(f"{path.name}: filename does not match candidate ID.")
            continue
        mismatches = _candidate_form_mismatches(
            candidate,
            expected[candidate_id],
        )
        if mismatches:
            errors.append(
                f"{path.name}: immutable candidate fields differ from source batch: "
                + ", ".join(mismatches)
            )
            continue
        forms_by_id[candidate_id] = form

    missing_ids = set(expected) - set(forms_by_id)
    errors.extend(f"Missing review form for {candidate_id}." for candidate_id in sorted(missing_ids))
    if source_page_texts is None:
        source_page_texts = load_source_page_texts(forms_by_id.values())

    approved = []
    decisions = []
    counts: Counter[str] = Counter()
    counts["pending"] += len(missing_ids)
    for candidate_id, form in forms_by_id.items():
        validation_errors = validate_decision_form(form, {
            "candidate_id": candidate_id,
            "proposed_answer": expected[candidate_id]["answer"],
        })
        review = form.get("review", {})
        decision = str(review.get("decision") or "PENDING").strip().upper()
        result = {
            "candidate_id": candidate_id,
            "decision": decision,
            "reviewer": str(review.get("reviewer") or "").strip(),
            "reason": str(review.get("reason") or "").strip(),
            "review_status": "PENDING_HUMAN_REVIEW",
            "validation_errors": validation_errors,
        }
        if validation_errors:
            counts["incomplete_or_invalid"] += 1
            result["decision"] = "PENDING"
            errors.extend(f"{candidate_id}: {error}" for error in validation_errors)
        elif decision == "PENDING":
            counts["pending"] += 1
        elif decision == "REJECT":
            result["review_status"] = "HUMAN_REJECTED"
            counts["rejected"] += 1
        elif decision == "EDIT_AND_RECHECK":
            result["review_status"] = "EDITED_PENDING_RECHECK"
            result["edited_answer"] = str(review.get("edited_answer") or "").strip()
            counts["edited_pending"] += 1
        elif decision == "APPROVE":
            approval_errors = validate_approved_form(form, source_page_texts)
            if approval_errors:
                counts["incomplete_or_invalid"] += 1
                result["decision"] = "PENDING"
                result["validation_errors"] = approval_errors
                errors.extend(f"{candidate_id}: {error}" for error in approval_errors)
            else:
                candidate = form["candidate"]
                result["review_status"] = "HUMAN_APPROVED_DETERMINISTIC_CHECKS_PASSED"
                result["question"] = str(candidate["question"])
                result["answer"] = str(
                    review.get("edited_answer")
                    or candidate["proposed_answer"]
                ).strip()
                result["evidence"] = str(candidate["supporting_passage"])
                result["source_filename"] = str(candidate["source_document"])
                result["page"] = int(candidate["pdf_page"])
                result["question_type"] = str(candidate["question_type"])
                result["legal_reference"] = str(
                    review.get("reviewed_legal_reference")
                    or candidate.get("legal_reference")
                    or ""
                ).strip()
                result["answer_supported_by_passage_confirmation"] = "YES"
                result["legal_reference_checked_against_original_source"] = "YES"
                approved.append(result)
                counts["approved"] += 1
        decisions.append(result)

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl_exclusive(output_dir / "review_decisions.jsonl", decisions)
    _write_jsonl_exclusive(output_dir / "approved_review_candidates.jsonl", approved)
    summary = {
        "input_batch_count": len(expected),
        "forms_found": len(forms_by_id),
        "reviewed_count": counts["approved"] + counts["rejected"] + counts["edited_pending"],
        "approved_count": counts["approved"],
        "rejected_count": counts["rejected"],
        "edited_pending_count": counts["edited_pending"],
        "pending_count": counts["pending"] + counts["incomplete_or_invalid"],
        "incomplete_or_invalid_count": counts["incomplete_or_invalid"],
        "errors": errors,
        "legal_correctness_claim": (
            "Human confirmations and deterministic checks are recorded, but this "
            "workflow does not independently establish legal correctness."
        ),
        "outputs": {
            "decisions": "review_decisions.jsonl",
            "approved_review_candidates": "approved_review_candidates.jsonl",
        },
    }
    _write_json_exclusive(output_dir / "review_import_summary.json", summary)
    return summary


def load_source_page_texts(
    forms: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], str]:
    requested: dict[str, set[int]] = {}
    for form in forms:
        candidate = form.get("candidate", {})
        try:
            page = int(candidate.get("pdf_page", ""))
        except (TypeError, ValueError):
            continue
        requested.setdefault(str(candidate.get("source_document") or ""), set()).add(page)

    result = {}
    for source, page_numbers in requested.items():
        pdf_path = SOURCE_PDFS.get(source)
        if pdf_path is None:
            continue
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Source PDF is missing: {pdf_path}")
        with pymupdf.open(pdf_path) as document:
            for page_number in page_numbers:
                if 1 <= page_number <= len(document):
                    result[(source, page_number)] = document[
                        page_number - 1
                    ].get_text("text", sort=True)
    return result


def _timestamped_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "reports" / f"{prefix}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--batch-csv", type=Path, default=STAGE15_DIR / "first_review_batch.csv")
    prepare.add_argument("--prioritized-csv", type=Path, default=STAGE15_DIR / "prioritized_candidates.csv")
    prepare.add_argument("--stage14-workspace", type=Path, default=STAGE14_DIR)
    prepare.add_argument("--output-dir", type=Path)
    importer = subparsers.add_parser("import")
    importer.add_argument("--forms-dir", type=Path, required=True)
    importer.add_argument("--batch-csv", type=Path, default=STAGE15_DIR / "first_review_batch.csv")
    importer.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        output = args.output_dir or _timestamped_dir("legal_qa_individual_review")
        summary = prepare_individual_review_workspace(
            args.batch_csv,
            args.prioritized_csv,
            args.stage14_workspace,
            output,
        )
    else:
        output = args.output_dir or _timestamped_dir("legal_qa_review_import")
        summary = process_review_forms(
            args.forms_dir,
            args.batch_csv,
            output,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
