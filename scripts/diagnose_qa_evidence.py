"""Non-destructive evidence matching diagnostic for existing QA candidates."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "fine_tuning" / "dataset"
CANDIDATE_FILES = {
    "train_candidate": DATASET_DIR / "expanded" / "legal_qa" / "train_candidates.jsonl",
    "validation_candidate": DATASET_DIR / "expanded" / "legal_qa" / "validation_candidates.jsonl",
}
EXISTING_FILE = DATASET_DIR / "qa_partial.jsonl"
OLD_CANDIDATE_AUDIT = (
    ROOT / "reports" / "qa_candidate_audit_20261004_final"
    / "pending_candidate_audit.csv"
)
SOURCE_PDFS = {
    "constitution": ROOT / "data" / "constitution_of_india.pdf",
    "consumer_protection": ROOT / "data" / "consumer_protection_act_2019.pdf",
}
OUTPUT_DIR = ROOT / "reports" / "qa_evidence_diagnostic_20261004_v3"
NEAR_DUPLICATE_THRESHOLD = 0.90
NEARBY_PAGE_RADIUS = 2
QUESTION_PATTERN = re.compile(
    r"Question:\s*(.*?)\s*\n\s*Legal evidence\s*:\s*",
    re.IGNORECASE | re.DOTALL,
)
REFERENCE_PATTERN = re.compile(
    r"\b(article|section|clause|part|chapter|schedule)\s+"
    r"([a-z0-9]+(?:\s*\([a-z0-9]+\))*)",
    re.IGNORECASE,
)
REQUIRED_FIELDS = (
    "instruction",
    "input",
    "output",
    "evidence",
    "source",
    "page",
    "legal_reference",
)


def normalize_text(value: Any) -> str:
    """Normalize common PDF layout artifacts without changing word order."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = (
        text.replace("\u00ad", "")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .casefold()
    )
    # PDF line wrapping commonly inserts a hyphen followed by a newline.
    text = re.sub(r"(?<=[a-z])-[ \t]*\r?\n[ \t]*(?=[a-z])", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def normalized_variants(value: Any) -> set[str]:
    """Retain both hyphen-joined and hyphen-separated extraction forms."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = (
        text.replace("\u00ad", "")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .casefold()
    )
    separated = " ".join(re.findall(r"[a-z0-9]+", text))
    joined = normalize_text(value)
    return {variant for variant in (joined, separated) if variant}


def read_jsonl(path: Path, prefix: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record_id = f"{prefix}-{line_number:04d}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                records.append({
                    "_record_id": record_id,
                    "_parse_error": f"Malformed JSON: {error.msg}",
                    "_raw_line": line.rstrip("\r\n"),
                })
                continue
            if not isinstance(value, dict):
                records.append({
                    "_record_id": record_id,
                    "_parse_error": "JSONL row is not a JSON object.",
                    "_raw_line": line.rstrip("\r\n"),
                })
                continue
            value["_record_id"] = record_id
            records.append(value)
    return records


def extract_question(record: dict[str, Any]) -> str:
    match = QUESTION_PATTERN.search(str(record.get("input", "")))
    return match.group(1).strip() if match else ""


def _page_hits(evidence: str, pages: list[str], indices: list[int]) -> list[int]:
    evidence_variants = normalized_variants(evidence)
    if not evidence_variants:
        return []
    return [
        index
        for index in indices
        if 0 <= index < len(pages)
        and any(
            evidence_variant in page_variant
            for evidence_variant in evidence_variants
            for page_variant in normalized_variants(pages[index])
        )
    ]


def match_evidence_pages(
    evidence: str,
    declared_page: Any,
    pages: list[str],
    nearby_radius: int = NEARBY_PAGE_RADIUS,
) -> dict[str, Any]:
    """Check declared page first, then adjacent pages without changing metadata."""
    try:
        page_number = int(declared_page)
    except (TypeError, ValueError):
        return {
            "match_status": "INVALID_DECLARED_PAGE",
            "declared_page_match": False,
            "nearby_page_matches": [],
            "matching_page_numbers": [],
        }
    if page_number < 1 or page_number > len(pages):
        return {
            "match_status": "INVALID_DECLARED_PAGE",
            "declared_page_match": False,
            "nearby_page_matches": [],
            "matching_page_numbers": [],
        }

    declared_index = page_number - 1
    declared_hits = _page_hits(evidence, pages, [declared_index])
    nearby_indices = [
        index
        for index in range(
            max(0, declared_index - nearby_radius),
            min(len(pages), declared_index + nearby_radius + 1),
        )
        if index != declared_index
    ]
    nearby_hits = _page_hits(evidence, pages, nearby_indices)

    # Evidence can be split across a PDF page break. Such a match is reported
    # as nearby and names both pages; the declared page is never rewritten.
    spanning_pairs = []
    evidence_variants = normalized_variants(evidence)
    pair_start = max(0, declared_index - nearby_radius)
    pair_end = min(len(pages) - 1, declared_index + nearby_radius)
    for left in range(pair_start, pair_end):
        right = left + 1
        if right >= len(pages):
            continue
        if (
            evidence_variants
            and not any(
                variant in page_variant
                for variant in evidence_variants
                for page_variant in normalized_variants(pages[left])
            )
            and not any(
                variant in page_variant
                for variant in evidence_variants
                for page_variant in normalized_variants(pages[right])
            )
            and any(
                variant
                in (
                    page_left_variant
                    + " "
                    + page_right_variant
                )
                for variant in evidence_variants
                for page_left_variant in normalized_variants(pages[left])
                for page_right_variant in normalized_variants(pages[right])
            )
        ):
            spanning_pairs.append([left + 1, right + 1])

    nearby_page_numbers = sorted({index + 1 for index in nearby_hits})
    matching_pages = sorted(
        set(nearby_page_numbers)
        | {number for pair in spanning_pairs for number in pair}
        | ({page_number} if declared_hits else set())
    )
    exact_page_hits = nearby_page_numbers + ([page_number] if declared_hits else [])
    if len(set(exact_page_hits)) > 1:
        status = "AMBIGUOUS_MATCH"
    elif declared_hits:
        status = "DECLARED_PAGE_MATCH"
    elif nearby_hits or spanning_pairs:
        status = "NEARBY_PAGE_MATCH"
    else:
        status = "NO_MATCH"

    return {
        "match_status": status,
        "declared_page_match": bool(declared_hits),
        "nearby_page_matches": nearby_page_numbers,
        "spanning_page_matches": spanning_pairs,
        "matching_page_numbers": matching_pages,
    }


def inspect_reference(reference: str, evidence: str) -> tuple[str, str]:
    normalized_ref = normalize_text(reference)
    normalized_evidence = normalize_text(evidence)
    if not normalized_ref:
        return "MISSING_REFERENCE", "Reference is empty; no correction was inferred."
    if normalized_ref == "not explicitly identified":
        return "NOT_ASSERTED", "No specific legal reference is asserted."

    reference_matches = list(REFERENCE_PATTERN.finditer(reference))
    claimed = [
        (match.group(1).casefold(), normalize_text(match.group(2)))
        for match in reference_matches
    ]
    if claimed:
        remainder = list(reference)
        for match in reference_matches:
            remainder[match.start():match.end()] = " " * (
                match.end() - match.start()
            )
        unclassified_words = set(
            re.findall(r"[a-z0-9]+", normalize_text("".join(remainder)))
        ) - {
            "and", "as", "act", "consumer", "constitution", "india",
            "of", "or", "per", "protection", "read", "the", "under",
            "with",
        }
        if unclassified_words:
            return "AMBIGUOUS_REFERENCE", (
                "The legal_reference field contains prose beyond recognizable "
                "identifiers; embedded identifiers are not treated as citations."
            )
        evidence_refs = {
            (kind.casefold(), normalize_text(number))
            for kind, number in REFERENCE_PATTERN.findall(evidence)
        }
        missing = [
            f"{kind} {number}"
            for kind, number in claimed
            if (kind, number) not in evidence_refs
        ]
        if missing:
            return "REFERENCE_CONFLICT_OR_UNCONFIRMED", (
                "Claimed identifier(s) not explicitly found in evidence: "
                + ", ".join(missing)
            )
        return "REFERENCE_MATCH", (
            "Every typed legal identifier in the reference appears in evidence."
        )

    if normalized_ref in normalized_evidence:
        # Text may occur but not form a recognizable legal identifier.
        if re.search(r"\bact\b", normalized_ref):
            return "REFERENCE_TEXT_MATCH_NEEDS_REVIEW", (
                "Act-name text appears in evidence; exact legal citation form "
                "requires human review."
            )
        return "AMBIGUOUS_REFERENCE", (
            "Reference text occurs, but its legal identifier type is unclear."
        )
    return "REFERENCE_CONFLICT_OR_UNCONFIRMED", (
        "Reference could not be explicitly matched in evidence."
    )


def _add_flag(flags: dict[str, set[str]], left: str, right: str) -> None:
    flags.setdefault(left, set()).add(right)
    flags.setdefault(right, set()).add(left)


def compare_records(
    candidates: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    question_by_id = {
        row["_record_id"]: normalize_text(extract_question(row))
        for row in candidates
    }
    answer_by_id = {
        row["_record_id"]: normalize_text(str(row.get("output", "")))
        for row in candidates
    }
    candidate_ids = set(question_by_id)
    for index, row in enumerate(existing, start=1):
        record_id = f"EXISTING-{index:04d}"
        question_by_id[record_id] = normalize_text(row.get("question", ""))
        answer_by_id[record_id] = normalize_text(row.get("answer", ""))

    exact_questions: dict[str, set[str]] = {}
    exact_answers: dict[str, set[str]] = {}
    near_questions: dict[str, set[str]] = {}
    identifiers = list(question_by_id)
    for left_index, left_id in enumerate(identifiers):
        for right_id in identifiers[left_index + 1:]:
            # Existing-only pairs are irrelevant to flagging candidates.
            if left_id not in candidate_ids and right_id not in candidate_ids:
                continue
            left_question = question_by_id[left_id]
            right_question = question_by_id[right_id]
            if left_question and left_question == right_question:
                _add_flag(exact_questions, left_id, right_id)
            left_answer = answer_by_id[left_id]
            right_answer = answer_by_id[right_id]
            if left_answer and left_answer == right_answer:
                _add_flag(exact_answers, left_id, right_id)
            if not left_question or not right_question or left_question == right_question:
                continue
            if abs(len(left_question) - len(right_question)) / max(
                len(left_question), len(right_question)
            ) > 1 - NEAR_DUPLICATE_THRESHOLD:
                continue
            matcher = SequenceMatcher(None, left_question, right_question)
            if (
                matcher.real_quick_ratio() >= NEAR_DUPLICATE_THRESHOLD
                and matcher.quick_ratio() >= NEAR_DUPLICATE_THRESHOLD
                and matcher.ratio() >= NEAR_DUPLICATE_THRESHOLD
            ):
                _add_flag(near_questions, left_id, right_id)
    return exact_questions, exact_answers, near_questions


def question_type(question: str) -> str:
    text = normalize_text(question)
    if re.search(r"\b(define|definition|what does .* mean|meaning of)\b", text):
        return "definition_candidate"
    if re.search(r"\b(how to|procedure|file a|appeal|remedy|complaint)\b", text):
        return "procedure_or_remedy_candidate"
    if re.search(r"\b(if|scenario|what happens when|under what conditions)\b", text):
        return "scenario_or_condition_candidate"
    if re.search(r"\b(compare|difference between|distinguish)\b", text):
        return "comparison_candidate"
    if re.search(r"\b(right|duty|obligation|entitled)\b", text):
        return "rights_or_duties_candidate"
    return "direct_or_other_candidate"


def read_previous_audit(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            row["record_id"]: row
            for row in csv.DictReader(stream)
            if row.get("record_id")
        }


def audit_candidate(
    record: dict[str, Any],
    source_pages: dict[str, list[str]],
    duplicate_flags: tuple[
        dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]
    ],
    prior_audit: dict[str, dict[str, str]],
) -> dict[str, Any]:
    exact_questions, exact_answers, near_questions = duplicate_flags
    record_id = record.get("_record_id", "UNKNOWN")
    question = extract_question(record)
    answer = str(record.get("output", "")).strip()
    evidence = str(record.get("evidence", "")).strip()
    source = str(record.get("source", "")).strip()
    reference = str(record.get("legal_reference", "")).strip()
    page = record.get("page", "")
    reasons = []
    invalid = []
    if record.get("_parse_error"):
        invalid.append(record["_parse_error"])
    missing_fields = [field for field in REQUIRED_FIELDS if field not in record]
    if missing_fields:
        invalid.append("Missing fields: " + ", ".join(missing_fields))
    for label, value in (
        ("question", question),
        ("answer", answer),
        ("evidence", evidence),
        ("source", source),
        ("instruction", record.get("instruction")),
    ):
        if not str(value or "").strip():
            invalid.append(f"Empty {label}.")
    if not question:
        invalid.append("Question could not be extracted from input.")
    pages = source_pages.get(source)
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        page_number = 0
    if pages is None:
        invalid.append(f"Unknown source label: {source or '<empty>'}.")
        match = {
            "match_status": "INVALID_SOURCE",
            "declared_page_match": False,
            "nearby_page_matches": [],
            "spanning_page_matches": [],
            "matching_page_numbers": [],
        }
    else:
        match = match_evidence_pages(evidence, page_number, pages)
        if match["match_status"] == "INVALID_DECLARED_PAGE":
            invalid.append(f"Invalid declared page: {page!r}.")
        elif match["match_status"] == "NO_MATCH":
            reasons.append("No normalized evidence match on declared or nearby pages.")
        elif match["match_status"] == "NEARBY_PAGE_MATCH":
            reasons.append(
                "Evidence appears on nearby page(s); declared citation is unchanged."
            )
        elif match["match_status"] == "AMBIGUOUS_MATCH":
            reasons.append("Evidence matches multiple pages; human review required.")
    if not page_number:
        invalid.append(f"Invalid declared page: {page!r}.")

    ref_status, ref_reason = inspect_reference(reference, evidence)
    if ref_status not in ("REFERENCE_MATCH", "NOT_ASSERTED"):
        reasons.append(ref_reason)
    normalized_answer = normalize_text(answer)
    normalized_evidence = normalize_text(evidence)
    answer_is_extract = bool(
        normalized_answer and normalized_answer in normalized_evidence
    )
    if answer and evidence and not answer_is_extract:
        reasons.append(
            "Answer is not a normalized extract; semantic support needs human review."
        )
    if record_id in exact_questions:
        reasons.append("Exact question duplicate flagged.")
    if record_id in exact_answers:
        reasons.append("Exact answer duplicate flagged.")
    if record_id in near_questions:
        reasons.append("Near-duplicate question flagged.")

    # Machine checks never promote candidate data to a legally validated state.
    status = "INVALID_RECORD" if invalid else "NEEDS_REVIEW"
    if invalid:
        reasons = invalid + reasons

    previous = prior_audit.get(str(record_id), {})
    question_candidates = {
        "question_mark": bool(question and question.endswith("?")),
        "question_type_heuristic": question_type(question),
    }
    return {
        "record_id": record_id,
        "candidate_split": str(record.get("split", "")),
        "question": question,
        "answer": answer,
        "source_document": source,
        "declared_page": page,
        "legal_reference": reference,
        "evidence": evidence,
        **question_candidates,
        "page_match_status": match["match_status"],
        "declared_page_match": match["declared_page_match"],
        "nearby_page_matches_diagnostic_only": ";".join(
            str(value) for value in match["nearby_page_matches"]
        ),
        "spanning_page_matches_diagnostic_only": json.dumps(
            match.get("spanning_page_matches", [])
        ),
        "matching_page_numbers_diagnostic_only": ";".join(
            str(value) for value in match["matching_page_numbers"]
        ),
        "reference_status": ref_status,
        "answer_is_normalized_extract": answer_is_extract,
        "exact_duplicate_question_ids": ";".join(
            sorted(exact_questions.get(str(record_id), set()))
        ),
        "near_duplicate_question_ids": ";".join(
            sorted(near_questions.get(str(record_id), set()))
        ),
        "exact_duplicate_answer_ids": ";".join(
            sorted(exact_answers.get(str(record_id), set()))
        ),
        "stage9_audit_status": previous.get("audit_status", "not_found"),
        "stage9_page_status": previous.get("page_evidence_status", "not_found"),
        "review_status": status,
        "review_reason": " ".join(reasons) or (
            "Needs human legal review; provenance checks alone cannot establish "
            "correctness."
        ),
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite diagnostic output directory: {OUTPUT_DIR}"
        )
    for path in (*CANDIDATE_FILES.values(), EXISTING_FILE, *SOURCE_PDFS.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    candidates = []
    for prefix, path in CANDIDATE_FILES.items():
        candidates.extend(read_jsonl(path, prefix))
    existing = read_jsonl(EXISTING_FILE, "existing")
    source_pages = {}
    for source, path in SOURCE_PDFS.items():
        document = pymupdf.open(path)
        try:
            source_pages[source] = [
                normalize_text(page.get_text("text")) for page in document
            ]
        finally:
            document.close()

    duplicate_sets = compare_records(candidates, existing)
    previous = read_previous_audit(OLD_CANDIDATE_AUDIT)
    rows = [
        audit_candidate(row, source_pages, duplicate_sets, previous)
        for row in candidates
    ]

    source_distribution = Counter(row["source_document"] or "<missing>" for row in rows)
    question_types = Counter(row["question_type_heuristic"] for row in rows)
    match_counts = Counter(row["page_match_status"] for row in rows)
    reference_counts = Counter(row["reference_status"] for row in rows)
    review_counts = Counter(row["review_status"] for row in rows)

    shortlist = [
        row for row in rows
        if row["stage9_audit_status"] == "NEEDS_REVIEW"
        and row["page_match_status"] in (
            "DECLARED_PAGE_MATCH",
            "NEARBY_PAGE_MATCH",
        )
        and row["reference_status"] == "REFERENCE_MATCH"
    ]
    shortlist.sort(
        key=lambda row: (
            row["page_match_status"] != "DECLARED_PAGE_MATCH",
            bool(row["exact_duplicate_question_ids"]),
            bool(row["near_duplicate_question_ids"]),
            bool(row["exact_duplicate_answer_ids"]),
            row["record_id"],
        )
    )

    output_csv = OUTPUT_DIR / "candidate_evidence_diagnostic.csv"
    output_shortlist = OUTPUT_DIR / "human_review_shortlist.csv"
    output_summary = OUTPUT_DIR / "diagnostic_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    for path, content_rows in (
        (output_csv, rows),
        (output_shortlist, shortlist),
    ):
        fields = list(content_rows[0]) if content_rows else [
            "record_id", "review_status", "review_reason"
        ]
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(content_rows)

    summary = {
        "scope": (
            "Diagnostic only. Nearby-page matches do not replace declared pages. "
            "No candidate is automatically accepted as legally correct."
        ),
        "candidate_files": {
            name: str(path.relative_to(ROOT))
            for name, path in CANDIDATE_FILES.items()
        },
        "existing_records_compared": {
            "path": str(EXISTING_FILE.relative_to(ROOT)),
            "count": len(existing),
        },
        "source_pdfs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "page_count": len(source_pages[name]),
            }
            for name, path in SOURCE_PDFS.items()
        },
        "candidate_count": len(rows),
        "review_status_counts": dict(review_counts),
        "page_match_status_counts": {
            status: match_counts.get(status, 0)
            for status in (
                "DECLARED_PAGE_MATCH",
                "NEARBY_PAGE_MATCH",
                "NO_MATCH",
                "AMBIGUOUS_MATCH",
                "INVALID_DECLARED_PAGE",
                "INVALID_SOURCE",
            )
        },
        "reference_status_counts": dict(reference_counts),
        "source_distribution": dict(source_distribution),
        "question_type_heuristic_distribution": dict(question_types),
        "duplicate_candidate_record_counts": {
            "exact_question": sum(bool(row["exact_duplicate_question_ids"]) for row in rows),
            "near_question": sum(bool(row["near_duplicate_question_ids"]) for row in rows),
            "exact_answer": sum(bool(row["exact_duplicate_answer_ids"]) for row in rows),
            "near_question_threshold": NEAR_DUPLICATE_THRESHOLD,
            "compared_against": "both candidate files and existing qa_partial.jsonl",
        },
        "human_review_shortlist_count": len(shortlist),
        "human_review_shortlist_rule": (
            "Stage 9 status NEEDS_REVIEW plus unique declared/nearby passage match "
            "and explicit reference match. Duplicate flags remain visible; no "
            "record is accepted or citation changed."
        ),
        "outputs": {
            "diagnostic": str(output_csv.relative_to(ROOT)),
            "shortlist": str(output_shortlist.relative_to(ROOT)),
            "summary": str(output_summary.relative_to(ROOT)),
        },
        "inputs_modified": False,
    }
    with output_summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
