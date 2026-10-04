"""Audit pending QA candidates against source PDFs without changing inputs."""

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
CANDIDATE_FILES = {
    "train_candidate": (
        DATASET_DIR / "expanded" / "legal_qa" / "train_candidates.jsonl"
    ),
    "validation_candidate": (
        DATASET_DIR / "expanded" / "legal_qa" / "validation_candidates.jsonl"
    ),
}
EXISTING_FILE = DATASET_DIR / "qa_partial.jsonl"
SOURCE_PDFS = {
    "constitution": ROOT / "data" / "constitution_of_india.pdf",
    "consumer_protection": ROOT / "data" / "consumer_protection_act_2019.pdf",
}
OUTPUT_DIR = ROOT / "reports" / "qa_candidate_audit_20261004_final"
NEAR_DUPLICATE_THRESHOLD = 0.90
REQUIRED_CANDIDATE_FIELDS = (
    "instruction",
    "input",
    "output",
    "evidence",
    "source",
    "page",
    "legal_reference",
)
QUESTION_PATTERN = re.compile(
    r"Question:\s*(.*?)\s*\n\s*Legal evidence\s*:\s*",
    re.IGNORECASE | re.DOTALL,
)
REFERENCE_SHAPE_PATTERN = re.compile(
    r"\b(?:article|section|clause|part|chapter|schedule)\s+"
    r"(?:\(?[a-z0-9ivxlcdm]+\)?)(?:\s*\([a-z0-9]+\))*\b"
    r"|\b(?:[a-z][a-z0-9&'()-]*\s+){1,8}act"
    r"(?:,?\s+\d{4}|\s+\d+\s+of\s+\d{4})?\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"\b(?:article|section|clause|part|chapter|schedule)\s+"
    r"[a-z0-9]+(?:\([a-z0-9]+\))*|\b\d+(?:[./-]\d+)*%?",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def read_jsonl(path: Path, prefix: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record_id = f"{prefix}-{line_number:04d}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                rows.append({
                    "_record_id": record_id,
                    "_parse_error": f"Malformed JSON: {error.msg}",
                    "_raw_line": line.rstrip("\r\n"),
                })
                continue
            if not isinstance(value, dict):
                rows.append({
                    "_record_id": record_id,
                    "_parse_error": "JSONL row is not a JSON object.",
                    "_raw_line": line.rstrip("\r\n"),
                })
                continue
            value["_record_id"] = record_id
            rows.append(value)
    return rows


def extract_question(record: dict[str, Any]) -> str:
    match = QUESTION_PATTERN.search(str(record.get("input", "")))
    return match.group(1).strip() if match else ""


def load_source_pages() -> dict[str, list[str]]:
    result = {}
    for source, path in SOURCE_PDFS.items():
        document = fitz.open(path)
        try:
            result[source] = [
                normalize_text(page.get_text("text"))
                for page in document
            ]
        finally:
            document.close()
    return result


def page_evidence_status(
    evidence: str,
    source: str,
    page: Any,
    source_pages: dict[str, list[str]],
) -> tuple[str, str]:
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        return "INVALID_PAGE_VALUE", "Page value is not an integer."
    if page_number < 1:
        return "INVALID_PAGE_VALUE", "Page numbers must be positive."
    pages = source_pages.get(source)
    if pages is None:
        return "UNKNOWN_SOURCE", "Source label does not map to a configured PDF."
    if page_number > len(pages):
        return "PAGE_OUT_OF_RANGE", (
            f"Declared page {page_number} exceeds PDF page count {len(pages)}."
        )
    normalized_evidence = normalize_text(evidence)
    if normalized_evidence and normalized_evidence in pages[page_number - 1]:
        return "MATCH", "Normalized evidence occurs on the declared source page."
    return "NO_MATCH", (
        "Evidence was not found on the declared page; this alone does not "
        "establish that the answer is legally wrong."
    )


def reference_status(reference: str, evidence: str) -> tuple[str, str]:
    normalized_reference = normalize_text(reference)
    normalized_evidence = normalize_text(evidence)
    if not normalized_reference:
        return "MISSING", "Reference is empty; do not infer or fill it automatically."
    if normalized_reference == "not explicitly identified":
        return "NOT_ASSERTED", "Candidate does not assert a specific reference."
    if normalized_reference in normalized_evidence:
        if not REFERENCE_SHAPE_PATTERN.search(reference):
            return "MALFORMED_OR_AMBIGUOUS", (
                "Reference text occurs in the evidence but does not have a "
                "recognizable legal identifier or Act citation form."
            )
        return "EXPLICIT_TEXT_MATCH", (
            "Normalized reference text occurs in the supplied evidence."
        )
    article_or_section = re.findall(
        r"\b(article|section)\s+(\d+[a-z]?)\b",
        reference,
        re.IGNORECASE,
    )
    if len(article_or_section) == 1:
        kind, number = article_or_section[0]
        number_in_evidence = bool(
            re.search(rf"\b{re.escape(number.casefold())}\b", normalized_evidence)
        )
        if number_in_evidence:
            return "TYPE_OR_CONTEXT_UNCONFIRMED", (
                f"Number {number} occurs, but the claimed {kind} type/context "
                "is not explicit in the evidence."
            )
    return "NOT_SUPPORTED_BY_EVIDENCE", (
        "The declared reference was not found in normalized evidence."
    )


def question_type(question: str) -> str:
    text = normalize_text(question)
    if re.search(r"\b(define|definition|what does .* mean|meaning of)\b", text):
        return "definition_candidate"
    if re.search(r"\b(how to|procedure|file a|appeal|remedy|complaint)\b", text):
        return "procedure_or_remedy_candidate"
    if re.search(r"\b(if|scenario|when .* happens|what happens when)\b", text):
        return "scenario_or_condition_candidate"
    if re.search(r"\b(compare|difference between|distinguish)\b", text):
        return "comparison_candidate"
    if re.search(r"\b(right|duty|obligation|entitled)\b", text):
        return "rights_or_duties_candidate"
    return "direct_or_other_candidate"


def normalize_answer(text: str) -> str:
    return normalize_text(text)


def _add_pair_flag(
    flags: dict[str, set[str]],
    left_id: str,
    right_id: str,
) -> None:
    flags.setdefault(left_id, set()).add(right_id)
    flags.setdefault(right_id, set()).add(left_id)


def compare_duplicates(
    candidates: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    questions = {
        row["_record_id"]: normalize_text(extract_question(row))
        for row in candidates
    }
    answers = {
        row["_record_id"]: normalize_answer(str(row.get("output", "")))
        for row in candidates
    }
    for index, row in enumerate(existing, start=1):
        record_id = f"EXISTING-{index:04d}"
        questions[record_id] = normalize_text(str(row.get("question", "")))
        answers[record_id] = normalize_answer(str(row.get("answer", "")))

    exact_questions: dict[str, set[str]] = {}
    exact_answers: dict[str, set[str]] = {}
    near_questions: dict[str, set[str]] = {}
    identifiers = list(questions)
    for left_index, left_id in enumerate(identifiers):
        left_question = questions[left_id]
        left_answer = answers[left_id]
        for right_id in identifiers[left_index + 1:]:
            right_question = questions[right_id]
            right_answer = answers[right_id]
            if left_question and left_question == right_question:
                _add_pair_flag(exact_questions, left_id, right_id)
            if left_answer and left_answer == right_answer:
                _add_pair_flag(exact_answers, left_id, right_id)
            if (
                left_question
                and right_question
                and left_question != right_question
                and SequenceMatcher(
                    None, left_question, right_question
                ).ratio() >= NEAR_DUPLICATE_THRESHOLD
            ):
                _add_pair_flag(near_questions, left_id, right_id)

    return exact_questions, exact_answers, near_questions


def audit_candidate(
    record: dict[str, Any],
    source_pages: dict[str, list[str]],
    exact_questions: dict[str, set[str]],
    exact_answers: dict[str, set[str]],
    near_questions: dict[str, set[str]],
) -> dict[str, Any]:
    record_id = str(record.get("_record_id", "UNKNOWN"))
    parse_error = str(record.get("_parse_error", ""))
    question = extract_question(record)
    answer = str(record.get("output", "")).strip()
    evidence = str(record.get("evidence", "")).strip()
    source = str(record.get("source", "")).strip()
    page = record.get("page", "")
    reference = str(record.get("legal_reference", "")).strip()
    reasons = []
    invalid_reasons = []

    if parse_error:
        invalid_reasons.append(parse_error)
    missing = [
        field for field in REQUIRED_CANDIDATE_FIELDS if field not in record
    ]
    if missing:
        invalid_reasons.append("Missing required field(s): " + ", ".join(missing))
    for label, value in (
        ("instruction", record.get("instruction")),
        ("input", record.get("input")),
        ("question", question),
        ("answer", answer),
        ("evidence", evidence),
        ("source", source),
    ):
        if not str(value or "").strip():
            invalid_reasons.append(f"Empty {label}.")
    if question and not question.endswith("?"):
        reasons.append("Question does not end with a question mark.")
    if question and len(question.split()) < 4:
        reasons.append("Question is unusually short and needs human review.")

    page_status, page_reason = page_evidence_status(
        evidence, source, page, source_pages
    )
    if page_status in ("INVALID_PAGE_VALUE", "UNKNOWN_SOURCE"):
        invalid_reasons.append(page_reason)
    elif page_status == "PAGE_OUT_OF_RANGE":
        invalid_reasons.append(page_reason)
    elif page_status == "NO_MATCH":
        reasons.append(page_reason)

    ref_status, ref_reason = reference_status(reference, evidence)
    if ref_status == "MISSING":
        reasons.append(ref_reason)
    elif ref_status in (
        "NOT_SUPPORTED_BY_EVIDENCE",
        "TYPE_OR_CONTEXT_UNCONFIRMED",
        "MALFORMED_OR_AMBIGUOUS",
    ):
        reasons.append(ref_reason)

    if answer and evidence:
        normalized_answer = normalize_answer(answer)
        normalized_evidence = normalize_text(evidence)
        answer_extract = bool(
            normalized_answer and normalized_answer in normalized_evidence
        )
        unsupported_numbers = sorted({
            normalize_text(match.group(0))
            for match in NUMBER_PATTERN.finditer(answer)
            if normalize_text(match.group(0))
            not in normalized_evidence
        })
        if unsupported_numbers:
            reasons.append(
                "Answer contains numeric/legal identifiers not found in evidence: "
                + ", ".join(unsupported_numbers)
            )
        if not answer_extract:
            reasons.append(
                "Answer is not an exact normalized extract; semantic support "
                "cannot be decided automatically and requires human review."
            )
    else:
        answer_extract = False
        unsupported_numbers = []

    if record_id in exact_questions:
        reasons.append("Exact question duplicate found in candidate/existing records.")
    if record_id in exact_answers:
        reasons.append("Exact answer duplicate found in candidate/existing records.")
    if record_id in near_questions:
        reasons.append(
            f"Near-duplicate question (similarity >= "
            f"{NEAR_DUPLICATE_THRESHOLD:.0%}) found in candidate/existing records."
        )

    if invalid_reasons:
        status = "INVALID_RECORD"
        reasons = invalid_reasons + reasons
    elif (
        page_status != "MATCH"
        or ref_status not in ("NOT_ASSERTED", "EXPLICIT_TEXT_MATCH")
        or not answer_extract
        or unsupported_numbers
        or record_id in exact_questions
        or record_id in exact_answers
        or record_id in near_questions
        or any(
            reason.startswith("Question does not end")
            or reason.startswith("Question is unusually short")
            for reason in reasons
        )
    ):
        status = "NEEDS_REVIEW"
    else:
        status = "PROVENANCE_MATCH"
        reasons.append(
            "Deterministic schema, page/evidence, reference, and extractive "
            "checks passed. Legal correctness is not established."
        )

    return {
        "record_id": record_id,
        "candidate_split": str(record.get("split", "")),
        "question": question,
        "answer": answer,
        "source_document": source,
        "declared_page": page,
        "legal_reference": reference,
        "evidence": evidence,
        "question_type_heuristic": question_type(question),
        "page_evidence_status": page_status,
        "reference_status": ref_status,
        "answer_exact_extract": answer_extract,
        "answer_numeric_or_identifier_flags": "; ".join(unsupported_numbers),
        "exact_duplicate_question_ids": "; ".join(
            sorted(exact_questions.get(record_id, set()))
        ),
        "exact_duplicate_answer_ids": "; ".join(
            sorted(exact_answers.get(record_id, set()))
        ),
        "near_duplicate_question_ids": "; ".join(
            sorted(near_questions.get(record_id, set()))
        ),
        "audit_status": status,
        "reason": " ".join(reasons),
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {OUTPUT_DIR}"
        )
    for path in (*CANDIDATE_FILES.values(), EXISTING_FILE, *SOURCE_PDFS.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit input is missing: {path}")

    candidate_records = []
    for prefix, path in CANDIDATE_FILES.items():
        candidate_records.extend(read_jsonl(path, prefix))
    existing_records = read_jsonl(EXISTING_FILE, "EXISTING")
    source_pages = load_source_pages()
    exact_q, exact_a, near_q = compare_duplicates(
        candidate_records, existing_records
    )
    audited = [
        audit_candidate(row, source_pages, exact_q, exact_a, near_q)
        for row in candidate_records
    ]

    statuses = Counter(row["audit_status"] for row in audited)
    candidate_sources = Counter(
        row["source_document"] or "<missing>" for row in audited
    )
    candidate_splits = Counter(
        row["candidate_split"] or "<missing>" for row in audited
    )
    question_types = Counter(
        row["question_type_heuristic"] for row in audited
    )
    page_statuses = Counter(row["page_evidence_status"] for row in audited)
    reference_statuses = Counter(row["reference_status"] for row in audited)
    output_csv = OUTPUT_DIR / "pending_candidate_audit.csv"
    output_summary = OUTPUT_DIR / "candidate_audit_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    with output_csv.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows(audited)

    summary = {
        "audit_scope": (
            "Deterministic schema, source/page, reference-text, extractive, "
            "numeric-token, and duplicate checks. PROVENANCE_MATCH does not "
            "mean legal correctness. NEEDS_REVIEW is not automatic rejection."
        ),
        "candidate_input_files": {
            label: str(path.relative_to(ROOT))
            for label, path in CANDIDATE_FILES.items()
        },
        "existing_records_compared": {
            "path": str(EXISTING_FILE.relative_to(ROOT)),
            "count": len(existing_records),
        },
        "source_pdfs": {
            source: {
                "path": str(path.relative_to(ROOT)),
                "page_count": len(source_pages[source]),
            }
            for source, path in SOURCE_PDFS.items()
        },
        "candidate_count": len(candidate_records),
        "status_counts": {
            status: statuses.get(status, 0)
            for status in (
                "PROVENANCE_MATCH",
                "NEEDS_REVIEW",
                "INVALID_RECORD",
            )
        },
        "candidate_split_counts": dict(candidate_splits),
        "source_distribution": dict(candidate_sources),
        "question_type_heuristic_distribution": dict(question_types),
        "page_evidence_status_counts": dict(page_statuses),
        "reference_status_counts": dict(reference_statuses),
        "unsupported_or_ambiguous_reference_records": sum(
            row["reference_status"] in (
                "NOT_SUPPORTED_BY_EVIDENCE",
                "TYPE_OR_CONTEXT_UNCONFIRMED",
                "MALFORMED_OR_AMBIGUOUS",
                "MISSING",
            )
            for row in audited
        ),
        "duplicate_record_counts": {
            "exact_question": sum(bool(row["exact_duplicate_question_ids"]) for row in audited),
            "exact_answer": sum(bool(row["exact_duplicate_answer_ids"]) for row in audited),
            "near_question": sum(bool(row["near_duplicate_question_ids"]) for row in audited),
            "near_question_similarity_threshold": NEAR_DUPLICATE_THRESHOLD,
        },
        "outputs": {
            "candidate_audit_csv": str(output_csv.relative_to(ROOT)),
            "summary_json": str(output_summary.relative_to(ROOT)),
        },
        "candidate_files_modified": False,
        "existing_records_modified": False,
    }
    with output_summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
