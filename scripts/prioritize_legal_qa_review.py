"""Rank pending legal QA candidates into a traceability-first review queue."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_source_grounded_qa_candidates import (  # noqa: E402
    HUMAN_REVIEW_STATUS,
    LEGAL_REFERENCE_RE,
    ROOT as PROJECT_ROOT,
    SOURCE_PDFS,
    normalize_text,
    reference_supported,
)

from scripts.legal_qa_human_review import (  # noqa: E402
    AUTHORITATIVE_RUN,
    _display_path,
)

DEFAULT_REVIEW_WORKSPACE = (
    PROJECT_ROOT / "reports" / "legal_qa_review_workspace_20261004T092435Z"
)
DEFAULT_BATCH_SIZE = 50
OUTPUT_COLUMNS = [
    "priority_rank",
    "candidate_id",
    "priority_score",
    "priority_reasons",
    "batch_selected",
    "batch_order",
    "recommended_action",
    "duplicate_group_id",
    "duplicate_candidate_ids",
    "duplicate_flags",
    "answer_template_repetition",
    "question",
    "answer",
    "evidence",
    "source_filename",
    "page",
    "legal_reference",
    "reference_status",
    "reference_confidence_flags",
    "reference_evidence_status",
    "evidence_status",
    "evidence_word_count",
    "evidence_quality_flags",
    "question_type",
    "review_status",
    "review_decision",
    "reviewer",
    "reason",
    "reviewer_notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object.")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_pdf_pages() -> dict[str, list[str]]:
    pages = {}
    for filename, pdf_path in SOURCE_PDFS.items():
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Source PDF is missing: {pdf_path}")
        with pymupdf.open(pdf_path) as document:
            pages[filename] = [
                normalize_text(page.get_text("text", sort=True))
                for page in document
            ]
    return pages


def evidence_status(
    row: Mapping[str, Any],
    source_pages: Mapping[str, Sequence[str]],
) -> str:
    evidence = normalize_text(row.get("evidence"))
    if not evidence:
        return "EMPTY_EVIDENCE"
    source = str(row.get("source_filename") or "")
    if source not in source_pages:
        return "UNKNOWN_SOURCE"
    try:
        page = int(row.get("page", ""))
    except (TypeError, ValueError):
        return "INVALID_PAGE"
    if page < 1 or page > len(source_pages[source]):
        return "INVALID_PAGE"
    return (
        "DECLARED_PAGE_MATCH"
        if evidence in source_pages[source][page - 1]
        else "NO_DECLARED_PAGE_MATCH"
    )


def reference_evidence_status(row: Mapping[str, Any]) -> str:
    reference = str(row.get("legal_reference") or "").strip()
    if not reference:
        return "UNCONFIRMED"
    if not LEGAL_REFERENCE_RE.fullmatch(reference):
        return "MALFORMED_REFERENCE"
    source = str(row.get("source_filename") or "")
    if not reference_supported(source, str(row.get("evidence") or ""), reference):
        return "REFERENCE_CONFLICT"
    if row.get("reference_status") != "EXPLICIT_NUMBERED_HEADING":
        return "REFERENCE_STATUS_CONFLICT"
    return "EXPLICIT_REFERENCE_CONSISTENT"


def _semicolon_items(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value or "").split(";") if item.strip()}


def annotate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    worksheet_rows: Sequence[Mapping[str, Any]],
    source_pages: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    worksheet_by_id = {
        str(row.get("candidate_id") or ""): row for row in worksheet_rows
    }
    candidate_ids = {str(row.get("candidate_id") or "") for row in candidates}
    if len(candidate_ids) != len(candidates) or "" in candidate_ids:
        raise ValueError("Candidate IDs must be nonempty and unique.")
    if set(worksheet_by_id) != candidate_ids:
        raise ValueError("Review worksheet IDs do not exactly match candidate IDs.")

    annotated = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        worksheet = worksheet_by_id[candidate_id]
        row = dict(candidate)
        row.update({
            field: worksheet.get(field, "")
            for field in (
                "duplicate_group_id",
                "duplicate_candidate_ids",
                "source_reference_conflict",
                "source_reference_conflict_reason",
                "reference_confidence_flags",
            )
        })
        row["duplicate_flags"] = sorted(
            _semicolon_items(candidate.get("duplicate_flags"))
            | _semicolon_items(worksheet.get("duplicate_flags"))
        )
        row["evidence_status"] = evidence_status(row, source_pages)
        row["reference_evidence_status"] = reference_evidence_status(row)
        normalized_evidence = normalize_text(row.get("evidence"))
        evidence_words = normalized_evidence.split()
        answer = normalize_text(row.get("answer"))
        evidence_flags = []
        if len(evidence_words) < 25:
            evidence_flags.append("SHORT_EVIDENCE_REVIEW")
        if row["evidence_status"] != "DECLARED_PAGE_MATCH":
            evidence_flags.append("SOURCE_PAGE_TRACEABILITY_CONCERN")
        if not answer or answer not in normalized_evidence:
            evidence_flags.append("ANSWER_NOT_FOUND_IN_EVIDENCE")
        if "REFERENCE_UNCONFIRMED" in _semicolon_items(
            row.get("reference_confidence_flags")
        ):
            evidence_flags.append("REFERENCE_UNCONFIRMED")
        row["evidence_word_count"] = len(evidence_words)
        row["evidence_quality_flags"] = evidence_flags
        row["answer_template_repetition"] = bool(
            {"REPEATED_EXACT_ANSWER", "REPEATED_ANSWER_TEMPLATE"}
            & set(row["duplicate_flags"])
        )
        row["review_status"] = candidate.get("review_status", "")
        row["review_decision"] = worksheet.get("review_decision", "")
        row["reviewer"] = worksheet.get("reviewer", "")
        row["reason"] = worksheet.get("reason", "")
        row["reviewer_notes"] = worksheet.get("reviewer_notes", "")
        if row["review_status"] != HUMAN_REVIEW_STATUS:
            raise ValueError(
                f"Candidate {candidate_id} is not pending human review."
            )
        if any(row[field] for field in (
            "review_decision",
            "reviewer",
            "reason",
            "reviewer_notes",
        )):
            raise ValueError(
                f"Candidate {candidate_id} has review values; prioritization "
                "must not fill reviewer decisions."
            )
        row["priority_score"], row["priority_reasons"] = score_candidate(row)
        row["recommended_action"] = (
            "REVIEW_REPRESENTATIVE_FIRST"
            if row["duplicate_candidate_ids"] or row["duplicate_group_id"]
            else "REVIEW"
        )
        annotated.append(row)
    return annotated


def score_candidate(row: Mapping[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if row.get("evidence_status") == "DECLARED_PAGE_MATCH":
        score += 50
        reasons.append("DECLARED_PAGE_EVIDENCE_MATCH")
    else:
        reasons.append("EVIDENCE_TRACEABILITY_NEEDS_REVIEW")
    if row.get("reference_evidence_status") == "EXPLICIT_REFERENCE_CONSISTENT":
        score += 35
        reasons.append("EXPLICIT_REFERENCE_CONSISTENT_WITH_PASSAGE")
    elif row.get("reference_evidence_status") == "UNCONFIRMED":
        reasons.append("REFERENCE_UNCONFIRMED")
    else:
        reasons.append("REFERENCE_CONFLICT_OR_FORMAT_CONCERN")
    if not row.get("answer_template_repetition"):
        score += 10
        reasons.append("NO_REPEATED_ANSWER_TEMPLATE_FLAG")
    else:
        reasons.append("REPEATED_ANSWER_TEMPLATE_REVIEW")
    duplicate_flags = set(row.get("duplicate_flags") or [])
    if "NEAR_QUESTION_DUPLICATE" in duplicate_flags:
        reasons.append("NEAR_DUPLICATE_GROUP_REVIEW_TOGETHER")
    if "EXACT_QUESTION_DUPLICATE" in duplicate_flags:
        reasons.append("EXACT_DUPLICATE_GROUP_REVIEW_TOGETHER")
    if "SHORT_EVIDENCE_REVIEW" not in set(row.get("evidence_quality_flags") or []):
        score += 5
        reasons.append("NONTRIVIAL_EVIDENCE_LENGTH")
    else:
        reasons.append("SHORT_EVIDENCE_REVIEW")
    return score, reasons


def _related_candidate_ids(row: Mapping[str, Any]) -> set[str]:
    return _semicolon_items(row.get("duplicate_candidate_ids"))


def select_first_review_batch(
    annotated: Sequence[Mapping[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Select deterministic diverse representatives, not duplicate group members."""
    if batch_size < 0:
        raise ValueError("batch_size cannot be negative.")
    if batch_size == 0:
        return []
    by_id = {str(row["candidate_id"]): row for row in annotated}
    if len(by_id) != len(annotated):
        raise ValueError("Candidate IDs must be unique.")

    group_members: dict[str, set[str]] = defaultdict(set)
    for row in annotated:
        candidate_id = str(row["candidate_id"])
        group_id = str(row.get("duplicate_group_id") or "")
        if group_id:
            group_members[group_id].add(candidate_id)
        for related_id in _related_candidate_ids(row):
            if related_id in by_id:
                group_members[candidate_id].add(related_id)
                group_members[related_id].add(candidate_id)

    # Select one strongest record per connected duplicate group for this first batch.
    visited = set()
    representative_ids = set()
    for candidate_id in sorted(by_id):
        if candidate_id in visited:
            continue
        stack = [candidate_id]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(group_members[current] - component)
        visited.update(component)
        representative = min(
            component,
            key=lambda member_id: (
                -int(by_id[member_id]["priority_score"]),
                member_id,
            ),
        )
        representative_ids.add(representative)

    remaining = [by_id[candidate_id] for candidate_id in representative_ids]
    source_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    selected = []

    def take_best(
        pool: Sequence[dict[str, Any]],
        predicate: Any,
    ) -> dict[str, Any] | None:
        matching = [row for row in pool if predicate(row)]
        if not matching:
            return None
        return min(
            matching,
            key=lambda row: (
                -int(row["priority_score"]),
                str(row["candidate_id"]),
            ),
        )

    # Seed the queue with the strongest available example from each source and
    # then each supported question type before filling by score.
    available_sources = sorted({
        str(row["source_filename"]) for row in remaining
    })
    available_types = sorted({
        str(row["question_type"]) for row in remaining
    })
    for source in available_sources:
        if len(selected) >= batch_size:
            break
        seed = take_best(
            remaining,
            lambda row, source=source: str(row["source_filename"]) == source,
        )
        if seed is not None:
            selected.append(seed)
            remaining.remove(seed)
            source_counts[source] += 1
            type_counts[str(seed["question_type"])] += 1
    for question_type in available_types:
        if len(selected) >= batch_size:
            break
        seed = take_best(
            remaining,
            lambda row, question_type=question_type: (
                str(row["question_type"]) == question_type
            ),
        )
        if seed is not None:
            selected.append(seed)
            remaining.remove(seed)
            source_counts[str(seed["source_filename"])] += 1
            type_counts[question_type] += 1

    representative_count = len(remaining) + len(selected)
    source_availability = Counter(
        str(row["source_filename"])
        for row in [*remaining, *selected]
    )
    while remaining and len(selected) < batch_size:
        selected_target = len(selected) + 1
        next_row = max(
            remaining,
            key=lambda row: (
                selected_target
                * source_availability[str(row["source_filename"])]
                / representative_count
                - source_counts[str(row["source_filename"])],
                selected_target / len(available_types)
                - type_counts[str(row["question_type"])],
                int(row["priority_score"]),
                str(row["candidate_id"]),
            ),
        )
        selected.append(dict(next_row))
        source_counts[str(next_row["source_filename"])] += 1
        type_counts[str(next_row["question_type"])] += 1
        remaining.remove(next_row)
    for index, row in enumerate(selected, start=1):
        row["batch_selected"] = True
        row["batch_order"] = index
        row["recommended_action"] = (
            "REVIEW_REPRESENTATIVE_FIRST"
            if row.get("duplicate_group_id") or row.get("duplicate_candidate_ids")
            else "REVIEW"
        )
        row["review_status"] = HUMAN_REVIEW_STATUS
        row["review_decision"] = ""
        row["reviewer"] = ""
        row["reason"] = ""
        row["reviewer_notes"] = ""
    return selected


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for field in OUTPUT_COLUMNS:
        value = row.get(field, "")
        if isinstance(value, (list, tuple, set)):
            result[field] = "; ".join(str(item) for item in value)
        elif field == "batch_selected":
            result[field] = "YES" if value else "NO"
        else:
            result[field] = value
    return result


def _write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field) or "UNSPECIFIED") for row in rows))


def _flag_distribution(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(Counter(
        flag
        for row in rows
        for flag in _semicolon_items(row.get(field))
    ))


def build_report(
    candidates: Sequence[Mapping[str, Any]],
    worksheet_rows: Sequence[Mapping[str, Any]],
    source_pages: Mapping[str, Sequence[str]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    annotated = annotate_candidates(candidates, worksheet_rows, source_pages)
    annotated.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            str(row["source_filename"]),
            str(row["question_type"]),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(annotated, start=1):
        row["priority_rank"] = index
        row["batch_selected"] = False
        row["batch_order"] = ""
    batch = select_first_review_batch(annotated, batch_size)
    batch_ids = {str(row["candidate_id"]) for row in batch}
    for row in annotated:
        if str(row["candidate_id"]) in batch_ids:
            row["batch_selected"] = True
            row["batch_order"] = next(
                item["batch_order"] for item in batch
                if item["candidate_id"] == row["candidate_id"]
            )
    batch.sort(key=lambda row: int(row["batch_order"]))
    summary = {
        "candidate_count": len(annotated),
        "selected_batch_count": len(batch),
        "batch_size_requested": batch_size,
        "batch_by_source": _distribution(batch, "source_filename"),
        "batch_by_question_type": _distribution(batch, "question_type"),
        "candidate_by_source": _distribution(annotated, "source_filename"),
        "candidate_by_question_type": _distribution(annotated, "question_type"),
        "candidate_by_source_and_question_type": dict(Counter(
            f"{row['source_filename']}|{row['question_type']}"
            for row in annotated
        )),
        "candidate_by_reference_status": _distribution(
            annotated,
            "reference_evidence_status",
        ),
        "candidate_by_evidence_status": _distribution(
            annotated,
            "evidence_status",
        ),
        "batch_by_evidence_status": _distribution(
            batch,
            "evidence_status",
        ),
        "batch_by_reference_status": _distribution(
            batch,
            "reference_evidence_status",
        ),
        "candidate_by_reference_confidence": _distribution(
            annotated,
            "reference_status",
        ),
        "duplicate_group_count": len({
            str(row["duplicate_group_id"])
            for row in annotated
            if row.get("duplicate_group_id")
        }),
        "candidate_rows_with_template_repetition": sum(
            bool(row["answer_template_repetition"]) for row in annotated
        ),
        "candidate_duplicate_flag_counts": _flag_distribution(
            annotated,
            "duplicate_flags",
        ),
        "candidate_evidence_quality_flag_counts": _flag_distribution(
            annotated,
            "evidence_quality_flags",
        ),
        "batch_duplicate_representatives": sum(
            bool(row.get("duplicate_group_id") or row.get("duplicate_candidate_ids"))
            for row in batch
        ),
        "review_status_counts": _distribution(annotated, "review_status"),
        "review_decision_count": sum(
            bool(row.get("review_decision")) for row in annotated
        ),
        "legal_correctness_claim": (
            "None. Page matching, heading/reference consistency, and deterministic "
            "selection aid review but do not establish legal correctness."
        ),
        "selection_method": (
            "Deterministic greedy ordering by traceability/reference score, with "
            "source and question-type balancing as tie-breakers; one representative "
            "per duplicate connected group is eligible for the first batch."
        ),
    }
    return annotated, batch, summary


def generate_prioritization_report(
    candidate_dir: Path,
    review_workspace: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    manifest = json.loads(
        (candidate_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("counts", {}).get("retained_candidates") != 1000:
        raise ValueError("Candidate run manifest does not declare 1,000 selected rows.")
    candidates = read_jsonl(candidate_dir / "candidates.jsonl")
    worksheet_manifest = json.loads(
        (review_workspace / "review_workspace_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if worksheet_manifest.get("candidate_count") != len(candidates):
        raise ValueError("Stage 14 workspace count does not match Stage 13 candidates.")
    worksheet_rows = read_csv(review_workspace / "review_worksheet.csv")
    annotated, batch, summary = build_report(
        candidates,
        worksheet_rows,
        load_pdf_pages(),
        batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv_exclusive(output_dir / "prioritized_candidates.csv", annotated)
    _write_csv_exclusive(output_dir / "first_review_batch.csv", batch)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_run": _display_path(candidate_dir),
        "review_workspace": _display_path(review_workspace),
        **summary,
        "outputs": {
            "all_candidates": "prioritized_candidates.csv",
            "first_review_batch": "first_review_batch.csv",
            "summary": "prioritization_summary.json",
        },
    }
    with (output_dir / "prioritization_summary.json").open(
        "x",
        encoding="utf-8",
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def _timestamped_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "reports" / f"legal_qa_review_prioritization_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=AUTHORITATIVE_RUN)
    parser.add_argument("--review-workspace", type=Path, default=DEFAULT_REVIEW_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=_timestamped_dir())
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 0:
        parser.error("--batch-size cannot be negative.")
    report = generate_prioritization_report(
        args.candidate_dir,
        args.review_workspace,
        args.output_dir,
        args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
