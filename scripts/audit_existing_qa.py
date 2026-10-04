"""Create a read-only provenance audit of the existing legal QA JSONL."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "fine_tuning" / "dataset"
OUTPUT_DIR = ROOT / "reports" / "qa_audit_20261004"
CANONICAL_FILE = DATASET_DIR / "qa_partial.jsonl"
SPLIT_FILES = {
    "train": DATASET_DIR / "qa_train.jsonl",
    "validation": DATASET_DIR / "qa_validation.jsonl",
    "test": DATASET_DIR / "qa_test.jsonl",
}
SOURCE_PDFS = {
    "constitution": ROOT / "data" / "constitution_of_india.pdf",
    "consumer_protection": ROOT / "data" / "consumer_protection_act_2019.pdf",
}
NEAR_DUPLICATE_THRESHOLD = 0.90
NOT_EXPLICIT = "not explicitly identified"
REQUIRED_FIELDS = (
    "question",
    "answer",
    "legal_reference",
    "supporting_passage",
    "source",
    "page",
)


def normalize_text(value: Any) -> str:
    """Normalize spacing and punctuation while preserving word order."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def load_pdf_pages() -> dict[str, list[str]]:
    pages: dict[str, list[str]] = {}
    for source, path in SOURCE_PDFS.items():
        document = fitz.open(path)
        try:
            pages[source] = [
                normalize_text(page.get_text("text"))
                for page in document
            ]
        finally:
            document.close()
    return pages


def locate_passage(passage: str, page_text: str) -> bool:
    normalized_passage = normalize_text(passage)
    normalized_page = normalize_text(page_text)
    return bool(normalized_passage and normalized_passage in normalized_page)


def inspect_reference(
    reference: str,
    source: str,
    supporting_passage: str,
) -> tuple[str, str]:
    normalized_reference = normalize_text(reference)
    normalized_passage = normalize_text(supporting_passage)

    if not normalized_reference:
        return "MISSING", "Legal reference is empty."
    if normalized_reference == NOT_EXPLICIT:
        return "NOT_ASSERTED", "Record does not assert a specific legal reference."
    if normalized_reference in normalized_passage:
        return "MATCHES_SUPPORTING_PASSAGE", (
            "Reference text occurs in the supplied supporting passage."
        )

    typed_references = re.findall(
        r"\b(article|section)\s+(\d+[a-z]?)\b",
        reference,
        flags=re.IGNORECASE,
    )
    if len(typed_references) == 1:
        kind, number = typed_references[0]
        has_reference_number = bool(
            re.search(rf"\b{re.escape(number.casefold())}\b", normalized_passage)
        )
        if has_reference_number and (
            kind.casefold() == "article" and source == "constitution"
            or kind.casefold() == "section"
            and source == "consumer_protection"
        ):
            return "NUMBER_PRESENT_TYPE_UNCONFIRMED", (
                "The number occurs in the passage, but the reference type "
                "is not explicitly established there."
            )

    if re.fullmatch(r"\d+[a-z]?", normalized_reference):
        return "AMBIGUOUS_UNTYPED_NUMBER", (
            "A bare number does not establish whether it is an Article, "
            "Section, or another numbered entry."
        )

    return "NOT_MATCHED", (
        "The declared legal reference was not verified in the supporting passage."
    )


def question_similarity_pairs(records: list[dict[str, Any]]) -> dict[int, list[int]]:
    normalized = [normalize_text(row.get("question")) for row in records]
    peers: dict[int, list[int]] = {index: [] for index in range(len(records))}
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if normalized[left] and normalized[right] and SequenceMatcher(
                None, normalized[left], normalized[right]
            ).ratio() >= NEAR_DUPLICATE_THRESHOLD:
                peers[left].append(right)
                peers[right].append(left)
    return peers


def duplicate_flags(
    records: list[dict[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    question_rows: dict[str, list[int]] = {}
    answer_rows: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        if question:
            question_rows.setdefault(question, []).append(index)
        if answer:
            answer_rows.setdefault(answer, []).append(index)

    question_duplicates = {index: [] for index in range(len(records))}
    answer_duplicates = {index: [] for index in range(len(records))}
    for indices in question_rows.values():
        if len(indices) > 1:
            for index in indices:
                question_duplicates[index] = [other for other in indices if other != index]
    for indices in answer_rows.values():
        if len(indices) > 1:
            for index in indices:
                answer_duplicates[index] = [other for other in indices if other != index]
    return (
        question_duplicates,
        answer_duplicates,
        question_similarity_pairs(records),
    )


def split_membership() -> dict[tuple[str, str], str]:
    memberships: dict[tuple[str, str], str] = {}
    for split, path in SPLIT_FILES.items():
        for row in load_jsonl(path):
            key = (
                normalize_text(row.get("question")),
                normalize_text(row.get("answer")),
            )
            memberships[key] = split
    return memberships


def prior_review_decisions() -> dict[str, str]:
    csv_path = DATASET_DIR / "qa_review_train_validation.csv"
    decisions_path = DATASET_DIR / "qa_review_decisions.json"
    if not csv_path.exists() or not decisions_path.exists():
        return {}

    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8-sig"))
    mapped = {}
    for index, row in enumerate(review_rows):
        decision = decisions.get(str(index), {}).get("decision", "unreviewed")
        key = normalize_text(row.get("question"))
        if key:
            mapped[key] = str(decision)
    return mapped


def audit_record(
    record: dict[str, Any],
    index: int,
    pages: dict[str, list[str]],
    split_by_question: dict[tuple[str, str], str],
    prior_reviews: dict[str, str],
    question_duplicates: dict[int, list[int]],
    answer_duplicates: dict[int, list[int]],
    near_duplicates: dict[int, list[int]],
) -> dict[str, Any]:
    reasons = []
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in record or record[field] is None
    ]
    empty_fields = [
        field
        for field in ("question", "answer", "supporting_passage", "source")
        if not str(record.get(field, "")).strip()
    ]
    source = str(record.get("source", "")).strip()
    page_value = record.get("page")
    try:
        page_number = int(page_value)
    except (TypeError, ValueError):
        page_number = 0

    fatal = []
    if missing_fields:
        fatal.append("Missing fields: " + ", ".join(missing_fields))
    if empty_fields:
        fatal.append("Empty required values: " + ", ".join(empty_fields))
    if source not in SOURCE_PDFS:
        fatal.append(f"Unknown source label: {source or '<empty>'}")
    if page_number < 1:
        fatal.append(f"Invalid page value: {page_value!r}")

    page_match = False
    page_status = "NOT_CHECKED"
    if not fatal:
        source_pages = pages[source]
        if page_number > len(source_pages):
            fatal.append(
                f"Page {page_number} is outside the {len(source_pages)}-page source."
            )
            page_status = "OUT_OF_RANGE"
        else:
            page_match = locate_passage(
                str(record.get("supporting_passage", "")),
                source_pages[page_number - 1],
            )
            page_status = "MATCH" if page_match else "NO_NORMALIZED_MATCH"
            if not page_match:
                reasons.append(
                    "Supporting passage did not match the declared PDF page; "
                    "this alone does not establish an incorrect answer."
                )

    reference_status, reference_reason = inspect_reference(
        str(record.get("legal_reference", "")),
        source,
        str(record.get("supporting_passage", "")),
    )
    reasons.append(reference_reason)
    if reference_status not in (
        "NOT_ASSERTED",
        "MATCHES_SUPPORTING_PASSAGE",
    ):
        reasons.append(
            "Legal-reference consistency requires manual source review."
        )

    answer = normalize_text(record.get("answer"))
    evidence = normalize_text(record.get("supporting_passage"))
    answer_is_extract = bool(answer and answer in evidence)
    if not answer_is_extract:
        reasons.append(
            "Answer is not an exact normalized extract of the supporting passage; "
            "semantic support requires manual review."
        )

    normalized_question = normalize_text(record.get("question"))
    normalized_answer = normalize_text(record.get("answer"))
    duplicate_question_ids = question_duplicates[index]
    duplicate_answer_ids = answer_duplicates[index]
    near_duplicate_ids = near_duplicates[index]
    if duplicate_question_ids:
        reasons.append("Exact duplicate question detected.")
    if duplicate_answer_ids:
        reasons.append("Exact duplicate answer detected.")
    if near_duplicate_ids:
        reasons.append(
            f"Near-duplicate question(s) at or above "
            f"{NEAR_DUPLICATE_THRESHOLD:.0%} similarity detected."
        )

    if fatal:
        status = "REJECTED"
        reasons = fatal + reasons
    elif (
        not page_match
        or reference_status not in ("NOT_ASSERTED", "MATCHES_SUPPORTING_PASSAGE")
        or not answer_is_extract
        or duplicate_question_ids
        or duplicate_answer_ids
        or near_duplicate_ids
    ):
        status = "NEEDS_REVIEW"
    else:
        status = "VERIFIED"
        reasons.append(
            "Schema, declared-page quotation, reference traceability, and "
            "extractive answer checks passed; question-answer legal correctness "
            "has not been established."
        )

    split_key = (normalized_question, normalized_answer)
    return {
        "record_id": f"QA-{index + 1:04d}",
        "source_split": split_by_question.get(split_key, "not_found_in_splits"),
        "question": str(record.get("question", "")),
        "answer": str(record.get("answer", "")),
        "source_document": source,
        "declared_page": page_value,
        "legal_reference": str(record.get("legal_reference", "")),
        "supporting_passage": str(record.get("supporting_passage", "")),
        "page_match": page_status,
        "reference_consistency": reference_status,
        "answer_is_extractive": answer_is_extract,
        "exact_duplicate_question_ids": ",".join(
            f"QA-{other + 1:04d}" for other in duplicate_question_ids
        ),
        "exact_duplicate_answer_ids": ",".join(
            f"QA-{other + 1:04d}" for other in duplicate_answer_ids
        ),
        "near_duplicate_question_ids": ",".join(
            f"QA-{other + 1:04d}" for other in near_duplicate_ids
        ),
        "previous_review_decision_not_validation": prior_reviews.get(
            normalized_question, "no_matching_prior_review"
        ),
        "audit_status": status,
        "reason": " ".join(reasons),
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit directory: {OUTPUT_DIR}"
        )
    records = load_jsonl(CANONICAL_FILE)
    pages = load_pdf_pages()
    question_duplicates, answer_duplicates, near_duplicates = duplicate_flags(
        records
    )
    split_by_question = split_membership()
    prior_reviews = prior_review_decisions()

    report = [
        audit_record(
            row,
            index,
            pages,
            split_by_question,
            prior_reviews,
            question_duplicates,
            answer_duplicates,
            near_duplicates,
        )
        for index, row in enumerate(records)
    ]
    status_counts = Counter(row["audit_status"] for row in report)
    source_counts = Counter(row["source_document"] for row in report)
    output_files = {
        "csv": OUTPUT_DIR / "existing_qa_audit.csv",
        "summary": OUTPUT_DIR / "audit_summary.json",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    with output_files["csv"].open(
        "x", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    summary = {
        "audit_scope": (
            "Deterministic schema, source/page traceability, citation-text, "
            "duplicate, and extractive-answer checks only. VERIFIED does not "
            "mean that a human has established legal correctness or that the "
            "question is correctly answered."
        ),
        "input_file": str(CANONICAL_FILE.relative_to(ROOT)),
        "split_files": {
            split: str(path.relative_to(ROOT))
            for split, path in SPLIT_FILES.items()
        },
        "source_pdfs": {
            source: str(path.relative_to(ROOT))
            for source, path in SOURCE_PDFS.items()
        },
        "records_audited": len(report),
        "status_counts": {
            status: status_counts.get(status, 0)
            for status in ("VERIFIED", "NEEDS_REVIEW", "REJECTED")
        },
        "source_distribution": dict(source_counts),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "duplicate_question_records": sum(
            bool(row["exact_duplicate_question_ids"]) for row in report
        ),
        "duplicate_answer_records": sum(
            bool(row["exact_duplicate_answer_ids"]) for row in report
        ),
        "near_duplicate_question_records": sum(
            bool(row["near_duplicate_question_ids"]) for row in report
        ),
        "prior_review_labels_are_not_used_as_validation": True,
        "outputs": {
            label: str(path.relative_to(ROOT))
            for label, path in output_files.items()
        },
    }
    with output_files["summary"].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
