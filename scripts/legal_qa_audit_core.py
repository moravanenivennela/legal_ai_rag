"""Read-only, source-text audit for existing legal QA candidate artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import pymupdf
except ImportError:  # pragma: no cover - exercised only in incomplete installs
    pymupdf = None


ROOT = Path(__file__).resolve().parents[1]
REPORT_PREFIX = "legal_qa_full_audit_"
NEAR_DUPLICATE_THRESHOLD = 0.92
QUESTION_STOPWORDS = frozenset({
    "a", "about", "according", "act", "an", "and", "are", "as", "at", "be",
    "by", "can", "could", "does", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "provide", "say", "state", "the", "to", "under",
    "what", "when", "where", "which", "who", "why", "with", "would",
})
STATUSES = (
    "SUPPORTED_CANDIDATE",
    "NEEDS_CORRECTION",
    "UNSUPPORTED_OR_MISMATCHED",
    "DUPLICATE",
    "INSUFFICIENT_SOURCE_EVIDENCE",
    "REQUIRES_HUMAN_LEGAL_REVIEW",
)
SOURCE_PDFS = {
    "constitution_of_india.pdf": "constitution_of_india.pdf",
    "consumer_protection_act_2019.pdf": "consumer_protection_act_2019.pdf",
}
REPORT_CSV_NAMES = {
    "first_pass_review.csv",
    "failure_pattern_analysis.csv",
    "suggested_corrections.csv",
    "suggestion_validation.csv",
    "corrected_review_batch.csv",
    "correction_change_log.csv",
    "ready_batch_audit.csv",
    "further_review_audit.csv",
    "first_review_batch.csv",
    "prioritized_candidates.csv",
    "review_worksheet.csv",
    "pending_candidate_audit.csv",
    "existing_qa_audit.csv",
    "candidate_evidence_diagnostic.csv",
    "human_review_shortlist.csv",
}
QUESTION_KEYS = ("original_question", "question", "Question", "query")
ANSWER_KEYS = (
    "original_answer",
    "answer",
    "output",
    "Expected",
    "expected_answer",
    "proposed_answer",
)
REFERENCE_KEYS = (
    "original_reference",
    "original_legal_reference",
    "candidate_legal_reference",
    "legal_reference",
    "reference",
    "Expected_Answer_Reference",
)
SOURCE_KEYS = (
    "source_pdf",
    "source_document",
    "source_filename",
    "Expected_Source",
    "source",
)
PAGE_KEYS = ("pdf_page_1_based", "pdf_page", "page", "expected_page")
EVIDENCE_KEYS = (
    "supporting_passage",
    "evidence",
    "exact_source_evidence_excerpt",
    "source_evidence_excerpt",
    "original_supporting_passage",
)
SUGGESTED_QUESTION_KEYS = ("suggested_question", "proposed_question")
SUGGESTED_ANSWER_KEYS = ("suggested_answer",)
SUGGESTED_REFERENCE_KEYS = (
    "suggested_reference",
    "suggested_legal_reference",
    "proposed_reference",
)
QUESTION_FROM_INPUT = re.compile(
    r"Question\s*:\s*(.*?)\s*(?:\n\s*Legal evidence\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
REFERENCE_RE = re.compile(
    r"\b(article|art\.?|section|sec\.?)\s*"
    r"(\d+[a-z]?)(?:\s*\(\s*([a-z0-9]+)\s*\))*",
    re.IGNORECASE,
)
PARAGRAPH_REFERENCE_RE = re.compile(
    r"\b(?:paragraph|para\.?)\s+(\d+[a-z]?)(?:\s*\(\s*([a-z0-9]+)\s*\))*",
    re.IGNORECASE,
)
ANSWER_NUMERIC_RE = re.compile(
    r"(?<![a-z])(?:\d{1,4}(?:[./-]\d{1,4})*%?)(?![a-z])",
    re.IGNORECASE,
)
QUALIFIER_RE = re.compile(
    r"\b(?:unless|except|provided|subject to|notwithstanding|only|"
    r"shall not|may not|not less than|not exceeding|at least|within)\b",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    return " ".join(TOKEN_RE.findall(text))


def _read_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    lowered = {str(key).casefold(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.casefold())
        if value is not None and str(value).strip():
            return value
    return ""


def _extract_candidate_rows(
    value: Any,
    source_path: str,
    row_number: int | str,
    source_kind: str,
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for index, item in enumerate(value, start=1):
            result.extend(
                _extract_candidate_rows(item, source_path, index, source_kind)
            )
        return result
    if not isinstance(value, Mapping):
        return []

    for collection_name in ("records", "candidates"):
        collection = value.get(collection_name)
        if isinstance(collection, list):
            result = []
            for index, item in enumerate(collection, start=1):
                result.extend(
                    _extract_candidate_rows(
                        item,
                        source_path,
                        f"{row_number}.{index}",
                        source_kind,
                    )
                )
            return result

    if isinstance(value.get("decisions"), list):
        result = []
        for index, decision in enumerate(value["decisions"], start=1):
            if not isinstance(decision, Mapping):
                continue
            candidate = decision.get("candidate_record") or decision.get("candidate")
            if isinstance(candidate, Mapping):
                item = dict(candidate)
                review = decision.get("review")
                if isinstance(review, Mapping):
                    item["_prior_decision"] = review.get("decision", "")
                else:
                    item["_prior_decision"] = decision.get("decision", "")
                result.extend(
                    _extract_candidate_rows(
                        item,
                        source_path,
                        f"{row_number}.{index}",
                        source_kind,
                    )
                )
        return result

    if isinstance(value.get("candidate"), Mapping):
        item = dict(value["candidate"])
        review = value.get("review")
        if isinstance(review, Mapping):
            item["_prior_decision"] = review.get("decision", "")
        return _extract_candidate_rows(item, source_path, row_number, source_kind)

    question = _read_value(value, QUESTION_KEYS)
    answer = _read_value(value, ANSWER_KEYS)
    if not question:
        raw_input = str(value.get("input", "") or "")
        match = QUESTION_FROM_INPUT.search(raw_input)
        if match:
            question = match.group(1).strip()
    if not question and not answer and not value.get("_parse_error"):
        return []

    occurrence = {
        "candidate_id": str(
            _read_value(value, ("candidate_id", "ID", "id")) or ""
        ).strip(),
        "question": str(question or "").strip(),
        "answer": str(answer or "").strip(),
        "reference": str(_read_value(value, REFERENCE_KEYS) or "").strip(),
        "source_document": str(_read_value(value, SOURCE_KEYS) or "").strip(),
        "page": str(_read_value(value, PAGE_KEYS) or "").strip(),
        "supporting_passage": str(
            _read_value(value, EVIDENCE_KEYS) or ""
        ).strip(),
        "suggested_question": str(
            _read_value(value, SUGGESTED_QUESTION_KEYS) or ""
        ).strip(),
        "suggested_answer": str(
            _read_value(value, SUGGESTED_ANSWER_KEYS)
            or (
                _read_value(value, ("proposed_answer",))
                if _read_value(value, ("original_answer",))
                else ""
            )
            or ""
        ).strip(),
        "suggested_reference": str(
            _read_value(value, SUGGESTED_REFERENCE_KEYS) or ""
        ).strip(),
        "prior_ai_recommendation": str(
            _read_value(value, ("recommendation", "first_pass_recommendation",
                                "ai_recommendation")) or ""
        ).strip(),
        "prior_decision": str(value.get("_prior_decision") or value.get(
            "review_decision", value.get("decision", "")
        )).strip(),
        "source_path": source_path,
        "source_row": str(row_number),
        "source_kind": source_kind,
        "parse_error": str(value.get("_parse_error") or ""),
    }
    if not occurrence["candidate_id"]:
        identity = "\0".join(
            normalize_text(occurrence[field])
            for field in (
                "question",
                "answer",
                "reference",
                "source_document",
                "page",
                "supporting_passage",
            )
        )
        occurrence["candidate_id"] = "auto-" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:20]
    return [occurrence]


def discover_candidate_files(root: Path) -> list[Path]:
    """Find known candidate and QA dataset formats, never model/index artifacts."""
    files: set[Path] = set()
    reports = root / "reports"
    if reports.is_dir():
        for path in reports.rglob("*"):
            if not path.is_file():
                continue
            if any(part.startswith(REPORT_PREFIX) for part in path.parts):
                continue
            if path.name == "candidates.jsonl" or path.name == "human_review_report.csv":
                files.add(path)
            elif path.suffix.lower() == ".csv" and path.name in REPORT_CSV_NAMES:
                files.add(path)
            elif (
                path.suffix.lower() == ".json"
                and path.parent.name in {
                    "candidate_forms",
                    "decision_forms",
                    "decision_records",
                }
            ):
                files.add(path)
            elif path.name in {
                "complete_review_export.json",
                "saved_review_selections.json",
            }:
                files.add(path)

    dataset = root / "fine_tuning" / "dataset"
    if dataset.is_dir():
        for path in dataset.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jsonl", ".csv"}:
                continue
            components = {part.casefold() for part in path.parts}
            name = path.name.casefold()
            if "classification" in components or "passage" in name:
                continue
            if "classifier" in name:
                continue
            if path.suffix.lower() == ".csv" and name not in {
                "qa_review_train_validation.csv",
                "legal_evaluation_500_candidates.csv",
            }:
                continue
            files.add(path)
    return sorted(files, key=lambda path: path.as_posix().casefold())


def load_candidate_occurrences(
    paths: Sequence[Path],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    occurrences: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        source_kind = path.suffix.lower().lstrip(".")
        try:
            if path.suffix.lower() == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    for index, row in enumerate(csv.DictReader(stream), start=2):
                        occurrences.extend(
                            _extract_candidate_rows(
                                row, relative_path, index, source_kind
                            )
                        )
            elif path.suffix.lower() == ".jsonl":
                with path.open(encoding="utf-8-sig") as stream:
                    for index, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as error:
                            occurrences.extend(_extract_candidate_rows(
                                {
                                    "candidate_id": "parse-" + hashlib.sha256(
                                        f"{relative_path}:{index}:{error.msg}".encode(
                                            "utf-8"
                                        )
                                    ).hexdigest()[:20],
                                    "_parse_error": (
                                        f"Malformed JSONL at line {index}: {error.msg}"
                                    )
                                },
                                relative_path,
                                index,
                                source_kind,
                            ))
                            continue
                        occurrences.extend(
                            _extract_candidate_rows(
                                value, relative_path, index, source_kind
                            )
                        )
            else:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                occurrences.extend(
                    _extract_candidate_rows(value, relative_path, 1, source_kind)
                )
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as error:
            errors.append({
                "source_path": relative_path,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return occurrences, errors


def _occurrence_priority(row: Mapping[str, Any]) -> tuple[int, str, str]:
    path = str(row.get("source_path", ""))
    if path.endswith(
        "qa_candidate_generation_stage13_20261004T091039Z_seed42/candidates.jsonl"
    ):
        rank = 0
    elif "candidate_generation" in path and path.endswith("candidates.jsonl"):
        rank = 1
    elif path.endswith("candidate_forms/" + path.split("/")[-1]):
        rank = 2
    elif path.startswith("fine_tuning/dataset/"):
        rank = 3
    else:
        rank = 4
    return rank, path.casefold(), str(row.get("source_row", ""))


def combine_candidate_occurrences(
    occurrences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[str(occurrence["candidate_id"])].append(occurrence)

    combined: list[dict[str, Any]] = []
    for candidate_id, versions in sorted(grouped.items()):
        ordered = sorted(versions, key=_occurrence_priority)

        def first_value(field: str) -> str:
            return next(
                (str(version.get(field) or "") for version in ordered
                 if str(version.get(field) or "").strip()),
                "",
            )

        question = first_value("question")
        answer = first_value("answer")
        reference = first_value("reference")
        source = first_value("source_document")
        page = first_value("page")
        evidence = first_value("supporting_passage")
        combined.append({
            "candidate_id": candidate_id,
            "question": question,
            "answer": answer,
            "reference": reference,
            "source_document": source,
            "page": page,
            "supporting_passage": evidence,
            "suggested_question": next(
                (str(v.get("suggested_question") or "") for v in ordered
                 if v.get("suggested_question")),
                "",
            ),
            "suggested_answer": next(
                (str(v.get("suggested_answer") or "") for v in ordered
                 if v.get("suggested_answer")),
                "",
            ),
            "suggested_reference": next(
                (str(v.get("suggested_reference") or "") for v in ordered
                 if v.get("suggested_reference")),
                "",
            ),
            "prior_ai_recommendations": sorted({
                str(v.get("prior_ai_recommendation") or "")
                for v in versions if v.get("prior_ai_recommendation")
            }),
            "prior_decisions": sorted({
                str(v.get("prior_decision") or "").upper()
                for v in versions if v.get("prior_decision")
            }),
            "source_occurrences": [
                {
                    "source_path": str(v.get("source_path", "")),
                    "source_row": str(v.get("source_row", "")),
                    "source_kind": str(v.get("source_kind", "")),
                    "candidate_id": str(v.get("candidate_id", "")),
                }
                for v in versions
            ],
            "source_record_count": len(versions),
            "source_field_variants": {
                field: list(dict.fromkeys(
                    str(v.get(field) or "")
                    for v in versions if str(v.get(field) or "").strip()
                ))
                for field in (
                    "question",
                    "answer",
                    "reference",
                    "source_document",
                    "page",
                    "supporting_passage",
                )
            },
            "input_errors": [
                str(v.get("parse_error"))
                for v in versions if v.get("parse_error")
            ],
        })
    return combined


def source_key(value: str) -> str | None:
    text = str(value or "").casefold().replace("\\", "/")
    basename = text.rsplit("/", 1)[-1]
    if "consumer" in text and ("protection" in text or "act" in text):
        return "consumer_protection_act_2019.pdf"
    if "constitution" in text:
        return "constitution_of_india.pdf"
    return SOURCE_PDFS.get(basename)


def _heading_index(
    pages_by_source: Mapping[str, Sequence[str] | None],
) -> dict[str, dict[str, set[int]]]:
    index: dict[str, dict[str, set[int]]] = {
        source: defaultdict(set) for source in SOURCE_PDFS
    }
    for source, pages in pages_by_source.items():
        if pages is None:
            continue
        for page_number, text in enumerate(pages, start=1):
            if source == "constitution_of_india.pdf":
                if re.search(r"\bcontents\b", text, re.IGNORECASE) or re.search(
                    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|"
                    r"eighth|ninth|tenth|eleventh|twelfth)\s+schedule\b",
                    text,
                    re.IGNORECASE,
                ):
                    continue
                pattern = re.compile(
                    r"(?m)^\s*(\d+[a-z]?)\.\s+[A-Z][^\n]{3,}",
                    re.IGNORECASE,
                )
            else:
                pattern = re.compile(
                    r"(?im)^\s*(\d{1,3})\.\s+[A-Z][^\n]{2,}"
                )
                if not pattern.search(text):
                    pattern = re.compile(
                        r"\b(\d{1,3})\.\s+In this Act\b",
                        re.IGNORECASE,
                    )
            for match in pattern.finditer(text):
                index[source][match.group(1).casefold()].add(page_number)
    return {source: dict(values) for source, values in index.items()}


def load_pdf_corpus(
    data_dir: Path,
) -> tuple[dict[str, list[str] | None], dict[str, str], dict[str, str]]:
    pages: dict[str, list[str] | None] = {}
    errors: dict[str, str] = {}
    paths: dict[str, str] = {}
    if pymupdf is None:
        return (
            {source: None for source in SOURCE_PDFS},
            {source: "PyMuPDF is not installed; source PDFs were not read."
             for source in SOURCE_PDFS},
            paths,
        )
    for source in SOURCE_PDFS:
        path = data_dir / source
        paths[source] = str(path)
        try:
            with pymupdf.open(path) as document:
                pages[source] = [
                    page.get_text("text", sort=True) for page in document
                ]
        except (OSError, RuntimeError, ValueError) as error:
            pages[source] = None
            errors[source] = f"{type(error).__name__}: {error}"
    return pages, errors, paths


def _page_number(value: Any) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _find_token_span(
    haystack: str,
    needle: str,
) -> tuple[int, int] | None:
    needle_tokens = list(TOKEN_RE.finditer(unicodedata.normalize("NFKC", needle).casefold()))
    haystack_tokens = list(TOKEN_RE.finditer(unicodedata.normalize("NFKC", haystack).casefold()))
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return None
    wanted = [match.group(0) for match in needle_tokens]
    words = [match.group(0) for match in haystack_tokens]
    width = len(wanted)
    for start in range(len(words) - width + 1):
        if words[start:start + width] == wanted:
            return haystack_tokens[start].start(), haystack_tokens[start + width - 1].end()
    return None


def _find_excerpt(
    text: str,
    query: str,
) -> str:
    span = _find_token_span(text, query)
    if span is None:
        return ""
    start, end = span
    while start > 0 and text[start - 1] in "\"'“‘([{":
        start -= 1
    while end < len(text) and text[end] in ".,;:!?—–-)]}\"'”’":
        end += 1
    return text[start:end]


def _reference_check(
    reference: str,
    source: str | None,
    page_number: int | None,
    pages: Sequence[str] | None,
    headings: Mapping[str, set[int]],
    supporting_passage: str = "",
) -> tuple[str, str, list[str]]:
    if not reference:
        return "MISSING_REFERENCE", "No explicit legal reference was supplied; none was inferred.", []
    if source is None or pages is None:
        return "SOURCE_UNAVAILABLE", "The source PDF could not be read.", []

    refs = list(REFERENCE_RE.finditer(reference))
    paragraph_refs = list(PARAGRAPH_REFERENCE_RE.finditer(reference))
    if not refs and not paragraph_refs:
        return "AMBIGUOUS_REFERENCE", "Reference text has no recognized article/section/paragraph identifier.", []
    if paragraph_refs and not refs:
        paragraph_number = paragraph_refs[0].group(1).casefold()
        context_pages = (
            range(max(1, (page_number or 1) - 1),
                  min(len(pages), (page_number or 1) + 1) + 1)
            if page_number is not None
            else range(0)
        )
        matching_pages = []
        paragraph_pattern = re.compile(
            rf"(?im)^\s*{re.escape(paragraph_number)}\.\s+"
        )
        for context_page in context_pages:
            text = pages[context_page - 1]
            is_schedule = bool(re.search(
                r"\b(?:first|second|third|fourth|fifth|sixth|seventh|"
                r"eighth|ninth|tenth|eleventh|twelfth)\s+schedule\b",
                text,
                re.IGNORECASE,
            ))
            if is_schedule and paragraph_pattern.search(text):
                matching_pages.append(context_page)
        if not matching_pages:
            return "REFERENCE_UNRESOLVED", (
                "The paragraph number could not be verified under a Schedule "
                "heading on or adjacent to the declared page."
            ), []
        subparts = re.findall(r"\(\s*([a-z0-9]+)\s*\)", reference, re.I)
        context_text = "\n".join(
            pages[context_page - 1] for context_page in context_pages
        )
        if any(
            not re.search(
                rf"(?<![a-z0-9])\(\s*{re.escape(part)}\s*\)",
                context_text,
                re.IGNORECASE,
            )
            for part in subparts
        ):
            return "PARTIAL_REFERENCE_SUPPORT", (
                "The Schedule paragraph is present, but one or more cited "
                "subparagraph markers were not found in the adjacent PDF text."
            ), matching_pages
        return "REFERENCE_MATCHED_PDF_HEADING", (
            "The cited Schedule paragraph and any listed subparagraph marker "
            "occur on or adjacent to the declared PDF page."
        ), matching_pages

    required_kind = (
        "article" if source == "constitution_of_india.pdf" else "section"
    )
    found_kinds = [
        match.group(1).casefold().rstrip(".")
        for match in refs
    ]
    if found_kinds and any(
        kind.startswith("art") != (required_kind == "article")
        for kind in found_kinds
    ):
        return "REFERENCE_TYPE_MISMATCH", (
            f"The source is {source}, but the reference type is not consistently "
            f"{required_kind.title()}."
        ), []
    if not found_kinds and not paragraph_refs:
        return "AMBIGUOUS_REFERENCE", "Could not identify the provision type.", []

    target_numbers = [
        match.group(2).casefold()
        for match in refs
    ]
    if not target_numbers and paragraph_refs:
        target_numbers = [paragraph_refs[0].group(1).casefold()]
    matched_pages: set[int] = set()
    for target in target_numbers:
        found = headings.get(target, set())
        if page_number is None:
            continue
        passage_has_target_heading = bool(re.search(
            rf"(?im)^\s*{re.escape(target)}\.\s+",
            supporting_passage,
        ))
        other_headings_on_page = {
            number
            for number, heading_pages in headings.items()
            if page_number in heading_pages
        }
        if (
            other_headings_on_page
            and target not in other_headings_on_page
            and not passage_has_target_heading
            and found
        ):
            return "REFERENCE_PAGE_MISMATCH", (
                f"The declared page contains a different numbered provision "
                f"heading ({', '.join(sorted(other_headings_on_page))}); "
                f"the cited heading {target} is not the supplied passage."
            ), sorted(found)
        if any(abs(found_page - page_number) <= 1 for found_page in found):
            matched_pages.update(
                found_page for found_page in found
                if abs(found_page - page_number) <= 1
            )
        elif not found:
            # OCR or a page continuation can prevent a reliable heading match.
            continue
        else:
            return "REFERENCE_PAGE_MISMATCH", (
                f"{required_kind.title()} {target} has a detected PDF heading, "
                "but not on or adjacent to the declared page."
            ), sorted(found)

    if matched_pages:
        subparts = re.findall(r"\(\s*([a-z0-9]+)\s*\)", reference, re.I)
        context_text = "\n".join(
            pages[context_page - 1]
            for context_page in range(
                max(1, (page_number or 1) - 1),
                min(len(pages), (page_number or 1) + 1) + 1,
            )
        )
        if any(
            not re.search(
                rf"(?<![a-z0-9])\(\s*{re.escape(part)}\s*\)",
                context_text,
                re.IGNORECASE,
            )
            for part in subparts
        ):
            return "PARTIAL_REFERENCE_SUPPORT", (
                "The main provision heading matches, but one or more cited "
                "subclause markers were not located in the adjacent PDF text."
            ), sorted(matched_pages)
        if paragraph_refs:
            paragraph_number = paragraph_refs[0].group(1).casefold()
            if page_number and 1 <= page_number <= len(pages):
                page_text = pages[page_number - 1]
                if not re.search(
                    rf"(?<![a-z0-9]){re.escape(paragraph_number)}\.",
                    page_text,
                    re.IGNORECASE,
                ):
                    return "PARTIAL_REFERENCE_SUPPORT", (
                        "The cited provision context is near the PDF page, but "
                        "the stated paragraph number was not found there."
                    ), sorted(matched_pages)
        return "REFERENCE_MATCHED_PDF_HEADING", (
            "The cited provision type and numbered heading match the source PDF "
            "on or adjacent to the declared page."
        ), sorted(matched_pages)

    return "REFERENCE_UNRESOLVED", (
        "The number could not be confidently mapped to a typed provision heading "
        "on or adjacent to the declared page."
    ), []


def _detect_issues(
    candidate: Mapping[str, Any],
    evidence_match: bool,
    reference_status: str,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    correction_options: list[str] = []
    question = str(candidate.get("question") or "")
    answer = str(candidate.get("answer") or "")
    evidence = str(candidate.get("supporting_passage") or "")

    if not question.strip():
        issues.append("MISSING_QUESTION")
    elif len(normalize_text(question).split()) < 4:
        issues.append("QUESTION_TOO_SHORT_OR_VAGUE")
    if question and not question.rstrip().endswith("?"):
        issues.append("QUESTION_PUNCTUATION")
        correction_options.append("question punctuation may need correction")
    if re.search(r"\b(?:all|every|each|complete|entire)\b", question, re.I):
        issues.append("BROAD_OR_COMPLETENESS_QUESTION")
    if re.search(r"\.\.\.|…|[\(\[]\s*$|\b(?:and|or|of|to|with|under)\s*$", answer, re.I):
        issues.append("POSSIBLY_TRUNCATED_ANSWER")
    if not answer.strip():
        issues.append("MISSING_ANSWER")
    if any(mark in question + answer + evidence for mark in ("\ufffd", "\u25a1")):
        issues.append("OCR_REPLACEMENT_CHARACTER")
    if re.search(r"\w\?\w|\?{2,}", "\n".join((question, answer, evidence))):
        issues.append("POSSIBLE_OCR_PUNCTUATION_ARTIFACT")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", question + answer + evidence):
        issues.append("CONTROL_CHARACTER_OR_EXTRACTION_ARTIFACT")
    if not evidence.strip():
        issues.append("MISSING_SUPPORTING_PASSAGE")
    elif not evidence_match:
        issues.append("SUPPORTING_PASSAGE_NOT_MATCHED_TO_DECLARED_PDF_PAGE")
    if reference_status in {
        "MISSING_REFERENCE",
        "AMBIGUOUS_REFERENCE",
        "REFERENCE_UNRESOLVED",
        "PARTIAL_REFERENCE_SUPPORT",
    }:
        issues.append(reference_status)

    if answer and evidence:
        answer_norm = normalize_text(answer)
        evidence_norm = normalize_text(evidence)
        answer_tokens = answer_norm.split()
        evidence_tokens = set(evidence_norm.split())
        evidence_ordered_tokens = evidence_norm.split()
        absent_numbers = []
        for match in ANSWER_NUMERIC_RE.finditer(answer):
            number_tokens = normalize_text(match.group(0)).split()
            if not number_tokens:
                continue
            if not all(token in evidence_tokens for token in number_tokens):
                absent_numbers.append(" ".join(number_tokens))
                continue
            width = len(number_tokens)
            if not any(
                evidence_ordered_tokens[index:index + width] == number_tokens
                for index in range(
                    len(evidence_ordered_tokens) - width + 1
                )
            ):
                absent_numbers.append(" ".join(number_tokens))
        absent_numbers = sorted(set(absent_numbers))
        if absent_numbers:
            issues.append("ANSWER_NUMERIC_OR_DATE_NOT_IN_EVIDENCE")
        if answer_norm and answer_norm not in evidence_norm:
            overlap = len(set(answer_tokens) & evidence_tokens) / max(
                1, len(set(answer_tokens))
            )
            if overlap < 0.20:
                issues.append("ANSWER_HAS_LOW_LEXICAL_SUPPORT")
            else:
                issues.append("ANSWER_IS_PARAPHRASE_REQUIRING_SEMANTIC_REVIEW")
        evidence_qualifiers = {
            normalize_text(match.group(0))
            for match in QUALIFIER_RE.finditer(evidence)
        }
        answer_qualifiers = {
            normalize_text(match.group(0))
            for match in QUALIFIER_RE.finditer(answer)
        }
        if evidence_qualifiers - answer_qualifiers:
            issues.append("POSSIBLE_OMITTED_QUALIFICATION")

    if question and not normalize_text(question):
        issues.append("QUESTION_HAS_NO_READABLE_TEXT")
    return sorted(set(issues)), correction_options


def audit_candidate(
    candidate: Mapping[str, Any],
    pages_by_source: Mapping[str, Sequence[str] | None],
    pdf_errors: Mapping[str, str] | None = None,
    headings_by_source: Mapping[str, Mapping[str, set[int]]] | None = None,
    pdf_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit one consolidated candidate using exact local PDF text only."""
    pdf_errors = pdf_errors or {}
    pdf_paths = pdf_paths or {}
    headings_by_source = headings_by_source or _heading_index(pages_by_source)
    source = source_key(str(candidate.get("source_document") or ""))
    page_number = _page_number(candidate.get("page"))
    pages = pages_by_source.get(source) if source else None
    reference = str(candidate.get("reference") or "").strip()
    evidence = str(candidate.get("supporting_passage") or "")
    question = str(candidate.get("question") or "")
    answer = str(candidate.get("answer") or "")
    issues: list[str] = []
    matched_evidence_page: int | None = None
    source_excerpt = ""
    answer_source_excerpt = ""
    page_count = len(pages) if pages is not None else 0

    if candidate.get("input_errors"):
        issues.extend("INPUT_PARSE_ERROR" for _ in candidate["input_errors"])
    if source is None:
        issues.append("UNKNOWN_SOURCE_DOCUMENT")
    if source is not None and pages is None:
        issues.append("SOURCE_PDF_MISSING_OR_UNREADABLE")
    if page_number is None:
        issues.append("MISSING_OR_INVALID_PAGE")
    elif pages is not None and page_number > len(pages):
        issues.append("PAGE_OUT_OF_RANGE")
    elif pages is not None and evidence:
        for location in (page_number, page_number - 1, page_number + 1):
            if 1 <= location <= len(pages):
                found = _find_excerpt(pages[location - 1], evidence)
                if found:
                    matched_evidence_page = location
                    source_excerpt = found
                    break
    if evidence and matched_evidence_page is None and pages is not None:
        issues.append("SOURCE_EVIDENCE_NOT_FOUND_IN_PDF_CONTEXT")
    if evidence and matched_evidence_page is not None:
        if matched_evidence_page != page_number:
            issues.append("SOURCE_EVIDENCE_PAGE_MISMATCH")
        answer_source_excerpt = _find_excerpt(
            source_excerpt,
            answer,
        )
    reference_status, reference_reason, reference_heading_pages = _reference_check(
        reference,
        source,
        page_number,
        pages,
        (headings_by_source or {}).get(source, {}) if source else {},
        evidence,
    )
    if reference_status != "REFERENCE_MATCHED_PDF_HEADING":
        issues.append(reference_status)

    secondary_issues, _ = _detect_issues(
        candidate,
        matched_evidence_page == page_number,
        reference_status,
    )
    issues.extend(secondary_issues)
    issues = sorted(set(issues))

    unsupported_mismatch = any(issue in issues for issue in (
        "PAGE_OUT_OF_RANGE",
        "REFERENCE_TYPE_MISMATCH",
        "REFERENCE_PAGE_MISMATCH",
        "ANSWER_NUMERIC_OR_DATE_NOT_IN_EVIDENCE",
        "ANSWER_HAS_LOW_LEXICAL_SUPPORT",
    ))
    insufficient = any(issue in issues for issue in (
        "UNKNOWN_SOURCE_DOCUMENT",
        "SOURCE_PDF_MISSING_OR_UNREADABLE",
        "MISSING_OR_INVALID_PAGE",
        "MISSING_QUESTION",
        "MISSING_ANSWER",
        "MISSING_SUPPORTING_PASSAGE",
        "SOURCE_EVIDENCE_NOT_FOUND_IN_PDF_CONTEXT",
        "SOURCE_EVIDENCE_PAGE_MISMATCH",
        "INPUT_PARSE_ERROR",
    ))
    question = str(candidate.get("question") or "")
    question_punctuation_fix = bool(
        question
        and not question.rstrip().endswith("?")
        and re.match(
            r"(?i)^\s*(?:what|when|where|which|who|whom|whose|why|how|"
            r"does|do|did|is|are|was|were|can|could|would|should|will)\b",
            question,
        )
        and not re.search(r"\w\?\w|\?{2,}", question)
    )
    punctuation_only_candidate = (
        matched_evidence_page == page_number
        and bool(answer)
        and bool(answer_source_excerpt)
        and normalize_text(answer) == normalize_text(answer_source_excerpt)
        and " ".join(answer.casefold().split())
        != " ".join(answer_source_excerpt.casefold().split())
        and not any(issue in issues for issue in (
            "OCR_REPLACEMENT_CHARACTER",
            "POSSIBLY_TRUNCATED_ANSWER",
            "ANSWER_NUMERIC_OR_DATE_NOT_IN_EVIDENCE",
        ))
    )
    serious_uncertainty = any(issue in issues for issue in (
        "QUESTION_TOO_SHORT_OR_VAGUE",
        "BROAD_OR_COMPLETENESS_QUESTION",
        "POSSIBLY_TRUNCATED_ANSWER",
        "OCR_REPLACEMENT_CHARACTER",
        "CONTROL_CHARACTER_OR_EXTRACTION_ARTIFACT",
        "ANSWER_IS_PARAPHRASE_REQUIRING_SEMANTIC_REVIEW",
        "POSSIBLE_OMITTED_QUALIFICATION",
        "MISSING_REFERENCE",
        "AMBIGUOUS_REFERENCE",
        "REFERENCE_UNRESOLVED",
        "PARTIAL_REFERENCE_SUPPORT",
        "REFERENCE_PAGE_MISMATCH",
    ))
    if (
        "POSSIBLE_OCR_PUNCTUATION_ARTIFACT" in issues
        and not punctuation_only_candidate
    ):
        serious_uncertainty = True
    if (
        "QUESTION_PUNCTUATION" in issues
        and not question_punctuation_fix
    ):
        serious_uncertainty = True
    fully_literal_match = bool(
        answer
        and evidence
        and normalize_text(answer) in normalize_text(evidence)
        and matched_evidence_page == page_number
    )

    if unsupported_mismatch:
        status = "UNSUPPORTED_OR_MISMATCHED"
        reason = "A source, page, reference, or answer-content mismatch was detected."
    elif insufficient:
        status = "INSUFFICIENT_SOURCE_EVIDENCE"
        reason = "Required source text or candidate fields could not be verified."
    elif (
        (punctuation_only_candidate or question_punctuation_fix)
        and not serious_uncertainty
    ):
        status = "NEEDS_CORRECTION"
        reason = (
            "Answer wording differs only in punctuation/formatting from the "
            "matching PDF excerpt; the source wording is offered as a suggestion."
        )
    elif serious_uncertainty:
        status = "REQUIRES_HUMAN_LEGAL_REVIEW"
        reason = (
            "Text checks found an ambiguity, possible omission, OCR issue, or "
            "non-literal answer that cannot safely be resolved automatically."
        )
    elif (
        matched_evidence_page == page_number
        and reference_status == "REFERENCE_MATCHED_PDF_HEADING"
        and fully_literal_match
        and not issues
    ):
        status = "SUPPORTED_CANDIDATE"
        reason = (
            "The supplied passage, numbered reference, page, and answer text "
            "match the provided PDF; this is source traceability only, not legal "
            "correctness or current-law verification."
        )
    elif matched_evidence_page is None or reference_status in {
        "SOURCE_UNAVAILABLE",
        "AMBIGUOUS_REFERENCE",
        "REFERENCE_UNRESOLVED",
    }:
        status = "INSUFFICIENT_SOURCE_EVIDENCE"
        reason = "The available source text does not settle the citation/evidence mapping."
    else:
        status = "REQUIRES_HUMAN_LEGAL_REVIEW"
        reason = (
            "Text traceability is partial, but semantic answer support and "
            "legal completeness cannot be established mechanically."
        )

    if punctuation_only_candidate:
        suggested_answer = answer_source_excerpt
        suggested_question = ""
        suggested_reference = ""
    else:
        suggested_answer = ""
        suggested_question = ""
        suggested_reference = ""
    if question_punctuation_fix:
        suggested_question = question.rstrip().rstrip(".;: ") + "?"

    prior_suggestion = {
        "question": str(candidate.get("suggested_question") or ""),
        "answer": str(candidate.get("suggested_answer") or ""),
        "reference": str(candidate.get("suggested_reference") or ""),
        "label": "EXISTING_SUGGESTION_NOT_VALIDATED_BY_THIS_AUTOMATION",
    }
    if not suggested_question:
        suggested_question = prior_suggestion["question"]
    if not suggested_answer:
        suggested_answer = prior_suggestion["answer"]
    if not suggested_reference:
        suggested_reference = prior_suggestion["reference"]

    pdf_error = pdf_errors.get(source, "") if source else ""
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "automated_status": status,
        "human_approval_status": "NOT_SET_BY_AUTOMATION",
        "source_text_support_is_not_legal_verification": True,
        "confidence": {
            "SUPPORTED_CANDIDATE": 0.90,
            "NEEDS_CORRECTION": 0.85,
            "UNSUPPORTED_OR_MISMATCHED": 0.90,
            "INSUFFICIENT_SOURCE_EVIDENCE": 0.95,
            "REQUIRES_HUMAN_LEGAL_REVIEW": 0.65,
        }.get(status, 0.90),
        "confidence_scope": "confidence in deterministic text classification only",
        "reason": reason,
        "issues": issues,
        "original_question": question,
        "original_answer": answer,
        "original_reference": reference,
        "source_document": source or str(candidate.get("source_document") or ""),
        "source_pdf_path": pdf_paths.get(source, "") if source else "",
        "declared_pdf_page": page_number,
        "source_pdf_page_count": page_count,
        "supporting_passage_supplied": evidence,
        "exact_source_evidence_excerpt": source_excerpt,
        "source_matched_answer_excerpt": answer_source_excerpt,
        "matched_source_pdf_page": matched_evidence_page,
        "reference_check_status": reference_status,
        "reference_check_reason": reference_reason,
        "reference_heading_pages": reference_heading_pages,
        "answer_exactly_supported_by_supplied_passage": fully_literal_match,
        "suggested_question": suggested_question,
        "suggested_answer": suggested_answer,
        "suggested_reference": suggested_reference,
        "suggestion_status": (
            "SUGGESTED_ONLY_NOT_APPROVED"
            if suggested_question or suggested_answer or suggested_reference
            else "NO_AUTOMATIC_CORRECTION"
        ),
        "existing_suggestion": prior_suggestion,
        "prior_ai_recommendations": list(
            candidate.get("prior_ai_recommendations") or []
        ),
        "prior_decisions_not_used_as_validation": list(
            candidate.get("prior_decisions") or []
        ),
        "source_occurrences": list(candidate.get("source_occurrences") or []),
        "source_record_count": int(candidate.get("source_record_count") or 1),
        "source_field_variants": dict(
            candidate.get("source_field_variants") or {}
        ),
        "duplicate_group_id": "",
        "duplicate_candidate_ids": [],
        "duplicate_relation": "",
        "preferred_duplicate_candidate_id": "",
        "preferred_candidate_reason": "",
        "pdf_read_error": pdf_error,
        "current_law_status": "NOT_CHECKED",
    }


def _candidate_quality(record: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    order = {
        "SUPPORTED_CANDIDATE": 0,
        "NEEDS_CORRECTION": 1,
        "REQUIRES_HUMAN_LEGAL_REVIEW": 2,
        "INSUFFICIENT_SOURCE_EVIDENCE": 3,
        "UNSUPPORTED_OR_MISMATCHED": 4,
    }
    return (
        order.get(str(record.get("automated_status")), 9),
        0 if record.get("exact_source_evidence_excerpt") else 1,
        0 if record.get("reference_check_status") == "REFERENCE_MATCHED_PDF_HEADING" else 1,
        0 if record.get("original_answer") else 1,
        str(record.get("candidate_id", "")),
    )


def detect_duplicate_groups(
    records: Sequence[dict[str, Any]],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[dict[str, Any]]:
    exact: dict[str, list[int]] = defaultdict(list)
    normalized_questions = [
        normalize_text(record.get("original_question"))
        for record in records
    ]
    for index, question in enumerate(normalized_questions):
        if question:
            exact[question].append(index)

    edges: dict[tuple[int, int], str] = {}
    for indices in exact.values():
        if len(indices) > 1:
            for left, right in zip(indices, indices[1:]):
                edges[(left, right)] = "EXACT_QUESTION"

    active = [
        index for index, text in enumerate(normalized_questions)
        if len(text) >= 20
    ]
    candidate_features: dict[int, set[str]] = {}
    feature_postings: dict[str, list[int]] = defaultdict(list)
    for index in active:
        tokens = normalized_questions[index].split()
        content = [
            token for token in tokens
            if token not in QUESTION_STOPWORDS and len(token) > 2
        ]
        features = {f"word:{token}" for token in content}
        features.update(
            f"pair:{content[offset]} {content[offset + 1]}"
            for offset in range(len(content) - 1)
        )
        candidate_features[index] = features
        for feature in features:
            feature_postings[feature].append(index)

    word_limit = max(40, int(len(active) * 0.02))
    pair_limit = max(60, int(len(active) * 0.04))
    feature_limits = {
        feature: word_limit if feature.startswith("word:") else pair_limit
        for feature in feature_postings
    }
    prefix_blocks: dict[str, list[int]] = defaultdict(list)
    suffix_blocks: dict[str, list[int]] = defaultdict(list)
    if len(active) > 700:
        for index in active:
            text = normalized_questions[index]
            prefix_blocks[text[:14]].append(index)
            suffix_blocks[text[-14:]].append(index)

    possible_pairs: set[tuple[int, int]] = set()
    if len(active) <= 700:
        for offset, left in enumerate(active):
            possible_pairs.update(
                (left, right) for right in active[offset + 1:]
            )
        candidate_pairs = possible_pairs
    else:
        candidate_pairs = None

    def compare_pair(left: int, right: int) -> None:
        if normalized_questions[left] == normalized_questions[right]:
            return
        left_text = normalized_questions[left]
        right_text = normalized_questions[right]
        short_length = min(len(left_text), len(right_text))
        long_length = max(len(left_text), len(right_text))
        if 2 * short_length / (short_length + long_length) < threshold:
            return
        matcher = SequenceMatcher(
            None,
            left_text,
            right_text,
            autojunk=False,
        )
        if matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold:
            edges[(left, right)] = "NEAR_QUESTION"

    if candidate_pairs is not None:
        for left, right in candidate_pairs:
            compare_pair(left, right)
    else:
        for left in active:
            neighbors: set[int] = set()
            for feature in candidate_features[left]:
                postings = feature_postings[feature]
                if 1 < len(postings) <= feature_limits[feature]:
                    neighbors.update(postings)
            for block in (prefix_blocks, suffix_blocks):
                postings = block[normalized_questions[left][:14]
                                 if block is prefix_blocks
                                 else normalized_questions[left][-14:]]
                if 1 < len(postings) <= 200:
                    neighbors.update(postings)
            for right in neighbors:
                if right > left:
                    compare_pair(left, right)

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    groups: list[dict[str, Any]] = []
    for members in components.values():
        if len(members) < 2:
            continue
        member_ids = [str(records[index]["candidate_id"]) for index in members]
        preferred_index = min(members, key=lambda index: _candidate_quality(records[index]))
        preferred_id = str(records[preferred_index]["candidate_id"])
        relations = {
            edges[(min(left, right), max(left, right))]
            for i, left in enumerate(members)
            for right in members[i + 1:]
            if (min(left, right), max(left, right)) in edges
        }
        group_id = "DUP-" + hashlib.sha256(
            "\0".join(sorted(member_ids)).encode("utf-8")
        ).hexdigest()[:12]
        preferred_reason = (
            "Preferred only for audit deduplication because it ranks best on "
            "source-PDF evidence match, citation/page traceability, and answer "
            "presence; this does not approve it."
        )
        groups.append({
            "duplicate_group_id": group_id,
            "duplicate_relation": (
                "EXACT_AND_NEAR_QUESTION" if len(relations) > 1
                else next(iter(relations), "NEAR_QUESTION")
            ),
            "similarity_threshold": threshold,
            "candidate_ids": sorted(member_ids),
            "preferred_candidate_id": preferred_id,
            "preferred_reason": preferred_reason,
            "members": [
                {
                    "candidate_id": str(records[index]["candidate_id"]),
                    "question": str(records[index].get("original_question") or ""),
                    "answer": str(records[index].get("original_answer") or ""),
                    "automated_status_before_duplicate_override": str(
                        records[index].get("automated_status")
                    ),
                    "preferred": index == preferred_index,
                    "relation_to_group": (
                        "PREFERRED_AUDIT_REPRESENTATIVE"
                        if index == preferred_index
                        else "DUPLICATE_NOT_DELETED"
                    ),
                }
                for index in members
            ],
        })
        for index in members:
            record = records[index]
            record["duplicate_group_id"] = group_id
            record["duplicate_candidate_ids"] = sorted(
                candidate_id for candidate_id in member_ids
                if candidate_id != str(record["candidate_id"])
            )
            record["duplicate_relation"] = (
                "PREFERRED_AUDIT_REPRESENTATIVE"
                if index == preferred_index
                else groups[-1]["duplicate_relation"]
            )
            record["preferred_duplicate_candidate_id"] = preferred_id
            record["preferred_candidate_reason"] = preferred_reason
            if index != preferred_index:
                record["automated_status"] = "DUPLICATE"
                record["reason"] = (
                    "Question is an exact or near duplicate of the preferred "
                    "audit representative. Original record is retained; this "
                    "is not a rejection or deletion."
                )
                record["confidence"] = 0.90
    return sorted(groups, key=lambda group: group["duplicate_group_id"])


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else [
        "candidate_id",
        "automated_status",
        "original_question",
        "original_answer",
        "original_reference",
        "source_document",
        "declared_pdf_page",
        "exact_source_evidence_excerpt",
        "issues",
        "reason",
    ]
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _status_subset(
    status: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matching = [record for record in records if record["automated_status"] == status]
    return {
        "schema_version": "1.0",
        "automated_status": status,
        "count": len(matching),
        "approval_notice": "Automated status is not human approval or legal verification.",
        "records": matching,
    }


def _refresh_suggestion_status(record: dict[str, Any]) -> None:
    original = {
        "question": str(record.get("original_question") or ""),
        "answer": str(record.get("original_answer") or ""),
        "reference": str(record.get("original_reference") or ""),
    }
    suggested = {
        "question": str(record.get("suggested_question") or ""),
        "answer": str(record.get("suggested_answer") or ""),
        "reference": str(record.get("suggested_reference") or ""),
    }

    def comparable_text(value: str) -> str:
        return " ".join(value.split()).casefold()

    changed_fields = [
        field
        for field, value in suggested.items()
        if value.strip()
        and comparable_text(value) != comparable_text(original[field])
    ]
    source_excerpt = str(record.get("source_matched_answer_excerpt") or "")
    record["suggested_fields_changed"] = changed_fields
    record["source_supported_correction_suggested"] = bool(
        suggested["answer"].strip()
        and suggested["answer"].strip() == source_excerpt.strip()
        and normalize_text(suggested["answer"])
        == normalize_text(original["answer"])
        and comparable_text(suggested["answer"])
        != comparable_text(original["answer"])
    )
    record["suggestion_status"] = (
        "SUGGESTED_ONLY_NOT_APPROVED"
        if changed_fields
        else "NO_AUTOMATIC_CORRECTION"
    )


def write_final_reports(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    duplicate_groups: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    import os
    import shutil

    staging_dir = output_dir / f".finalizing_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    staging_dir.mkdir(exist_ok=False)
    normalized_records = [dict(record) for record in records]
    for record in normalized_records:
        _refresh_suggestion_status(record)
    records = normalized_records
    full = {
        "schema_version": "1.0",
        "generated_at_utc": _now(),
        "audit_scope": (
            "Deterministic local PDF text, page, reference-heading, answer-text, "
            "format, and duplicate checks. No legal opinion or current-law "
            "verification is provided."
        ),
        "count_unique_candidates": len(records),
        "records": list(records),
    }
    _write_json(staging_dir / "full_audit.json", full)
    _write_csv(staging_dir / "full_audit.csv", records)
    _write_json(
        staging_dir / "supported_candidates.json",
        _status_subset("SUPPORTED_CANDIDATE", records),
    )
    _write_json(
        staging_dir / "needs_correction.json",
        _status_subset("NEEDS_CORRECTION", records),
    )
    _write_json(
        staging_dir / "unsupported_candidates.json",
        _status_subset("UNSUPPORTED_OR_MISMATCHED", records),
    )
    _write_json(staging_dir / "duplicates.json", {
        "schema_version": "1.0",
        "count_duplicate_groups": len(duplicate_groups),
        "groups": list(duplicate_groups),
        "original_records_deleted": False,
        "preferred_means_audit_representative_only": True,
    })
    human_rows = [
        record for record in records
        if record["automated_status"] in {
            "INSUFFICIENT_SOURCE_EVIDENCE",
            "REQUIRES_HUMAN_LEGAL_REVIEW",
        }
    ]
    _write_json(staging_dir / "human_review_required.json", {
        "schema_version": "1.0",
        "count": len(human_rows),
        "approval_notice": "These records remain unresolved; no approval is inferred.",
        "records": human_rows,
    })
    status_counts = Counter(str(record["automated_status"]) for record in records)
    issue_counts = Counter(
        issue for record in records for issue in record.get("issues", [])
    )
    source_counts = Counter(
        str(record.get("source_document") or "UNKNOWN")
        for record in records
    )
    metrics = dict(summary)
    metrics.update({
        "unique_candidates_processed": len(records),
        "status_counts": {status: status_counts.get(status, 0) for status in STATUSES},
        "supported_candidates": status_counts.get("SUPPORTED_CANDIDATE", 0),
        "corrections_suggested": sum(
            record.get("suggestion_status") == "SUGGESTED_ONLY_NOT_APPROVED"
            for record in records
        ),
        "source_supported_corrections_suggested": sum(
            record.get("source_supported_correction_suggested", False)
            for record in records
        ),
        "existing_suggestions_with_changed_wording": sum(
            any(
                " ".join(str(
                    record.get("existing_suggestion", {}).get(field) or ""
                ).split()).casefold()
                != " ".join(str(
                    record.get(original_field) or ""
                ).split()).casefold()
                and bool(str(
                    record.get("existing_suggestion", {}).get(field) or ""
                ).strip())
                for field, original_field in (
                    ("question", "original_question"),
                    ("answer", "original_answer"),
                    ("reference", "original_reference"),
                )
            )
            for record in records
        ),
        "unsupported_records": status_counts.get("UNSUPPORTED_OR_MISMATCHED", 0),
        "duplicate_records": status_counts.get("DUPLICATE", 0),
        "duplicate_groups": len(duplicate_groups),
        "unresolved_references": sum(
            issue_counts.get(issue, 0)
            for issue in (
                "MISSING_REFERENCE",
                "AMBIGUOUS_REFERENCE",
                "REFERENCE_UNRESOLVED",
                "PARTIAL_REFERENCE_SUPPORT",
            )
        ),
        "source_evidence_failures": sum(
            issue_counts.get(issue, 0)
            for issue in (
                "SOURCE_PDF_MISSING_OR_UNREADABLE",
                "SOURCE_EVIDENCE_NOT_FOUND_IN_PDF_CONTEXT",
                "SOURCE_EVIDENCE_PAGE_MISMATCH",
                "MISSING_SUPPORTING_PASSAGE",
            )
        ),
        "issue_counts": dict(sorted(issue_counts.items())),
        "source_counts": dict(source_counts),
        "human_approved_count_by_this_automation": 0,
        "legally_verified_count_by_this_automation": 0,
        "accuracy_precision_recall_f1": "NOT_MEASURED",
        "near_duplicate_detection_method": (
            "Exact normalized questions are grouped across all identities. "
            "Near-duplicate candidates use rare content-word pairs and question "
            "prefix/suffix blocking, followed by SequenceMatcher at the recorded "
            "threshold. Synonym-based or semantically similar questions may not "
            "be detected."
        ),
    })
    _write_json(staging_dir / "audit_metrics.json", metrics)
    _write_json(staging_dir / "summary_report.json", metrics)
    _write_summary_markdown(staging_dir / "summary_report.md", metrics)
    for path in staging_dir.iterdir():
        os.replace(path, output_dir / path.name)
    shutil.rmtree(staging_dir)


def _write_summary_markdown(path: Path, metrics: Mapping[str, Any]) -> None:
    counts = metrics.get("status_counts", {})
    lines = [
        "# Full legal QA candidate audit",
        "",
        f"- Unique candidate identities audited: {metrics.get('unique_candidates_processed', 0)}",
        f"- Source occurrences consolidated: {metrics.get('source_occurrences_loaded', 0)}",
        f"- Input files discovered: {metrics.get('input_file_count', 0)}",
        f"- Input parsing/read errors: {metrics.get('input_error_count', 0)}",
        f"- Candidate processing errors: {metrics.get('candidate_error_count', 0)}",
        "",
        "## Automated classification counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{status}` | {counts.get(status, 0)} |" for status in STATUSES)
    lines.extend([
        "",
        "## Interpretation and limits",
        "",
        "These classifications describe deterministic source-text, page, "
        "reference-heading, formatting, and duplicate checks against the supplied "
        "PDFs. A `SUPPORTED_CANDIDATE` means only that the supplied text is "
        "traceable under those checks. It does **not** establish that the answer "
        "is legally correct, complete, current, or suitable for training.",
        "",
        "Prior AI recommendations and any saved dropdown selections are recorded "
        "separately and are not used as legal evidence or approval. No candidate "
        "was approved or legally verified by this automation. Existing reviewer "
        "decisions, if any, remain unchanged in their original files.",
        "",
        "Suggested wording is labeled `SUGGESTED_ONLY_NOT_APPROVED`; original "
        "question, answer, reference, source evidence, and source occurrence "
        "metadata remain preserved in the full audit. Duplicate preference means "
        "audit representative only; no duplicate source record was deleted.",
        "",
        "No accuracy, precision, recall, or F1 score was calculated because no "
        "independently labeled evaluation set was used.",
        "",
        "Near-duplicate matching uses a deterministic blocking pass before the "
        "similarity threshold to keep large candidate sets tractable. It can miss "
        "paraphrases or questions with little shared wording; duplicate groups "
        "are audit flags, not a claim that all semantic duplicates were found.",
        "",
        "See `audit_metrics.json` for issue counts, source distribution, "
        "unresolved references, and evidence-failure counts.",
        "",
    ])
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def _input_fingerprint(paths: Sequence[Path], root: Path, data_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted([*paths, *(data_dir / source for source in SOURCE_PDFS)]):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = str(path.resolve())
        digest.update(relative.encode("utf-8"))
        if not path.is_file():
            digest.update(b"\0MISSING")
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_full_audit(
    root: Path = ROOT,
    data_dir: Path | None = None,
    output_root: Path | None = None,
    batch_size: int = 100,
    resume_from: Path | None = None,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    data_dir = data_dir or root / "data"
    output_root = output_root or root / "reports"
    inputs = discover_candidate_files(root)
    input_occurrences, input_errors = load_candidate_occurrences(inputs, root)
    candidates = combine_candidate_occurrences(input_occurrences)
    if resume_from is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_root / f"{REPORT_PREFIX}{stamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir = resume_from
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {output_dir}")
        manifest = output_dir / "run_manifest.json"
        if manifest.exists() and json.loads(
            manifest.read_text(encoding="utf-8")
        ).get("completed"):
            raise FileExistsError(
                "This audit folder is already complete; choose a new run instead "
                "of overwriting completed output."
            )
    fingerprint = _input_fingerprint(inputs, root, data_dir)
    checkpoints_dir = output_dir / ".checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": _now(),
        "root": str(root.resolve()),
        "data_dir": str(data_dir.resolve()),
        "input_fingerprint": fingerprint,
        "input_file_count": len(inputs),
        "source_occurrences_loaded": len(input_occurrences),
        "unique_candidate_count": len(candidates),
        "batch_size": batch_size,
        "input_files": [
            path.relative_to(root).as_posix() for path in inputs
        ],
        "completed": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "input_fingerprint",
            "unique_candidate_count",
            "batch_size",
        ):
            if previous.get(key) != manifest.get(key):
                raise ValueError(
                    f"Cannot resume: {key} changed since the audit began."
                )
    else:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    pages_by_source, pdf_errors, pdf_paths = load_pdf_corpus(data_dir)
    headings = _heading_index(pages_by_source)
    total_batches = (len(candidates) + batch_size - 1) // batch_size
    audited_by_id: dict[str, dict[str, Any]] = {}
    for batch_index in range(total_batches):
        start = batch_index * batch_size
        batch_candidates = candidates[start:start + batch_size]
        checkpoint_path = checkpoints_dir / f"batch_{batch_index + 1:05d}.json"
        if checkpoint_path.exists():
            batch_results = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(batch_results, list) or len(batch_results) != len(batch_candidates):
                raise ValueError(f"Invalid resume checkpoint: {checkpoint_path}")
        else:
            batch_results = []
            for candidate in batch_candidates:
                try:
                    batch_results.append(audit_candidate(
                        candidate,
                        pages_by_source,
                        pdf_errors,
                        headings,
                        pdf_paths,
                    ))
                except Exception as error:
                    batch_results.append({
                        "candidate_id": candidate["candidate_id"],
                        "automated_status": "INSUFFICIENT_SOURCE_EVIDENCE",
                        "human_approval_status": "NOT_SET_BY_AUTOMATION",
                        "source_text_support_is_not_legal_verification": True,
                        "confidence": 0.95,
                        "confidence_scope": "confidence in deterministic text classification only",
                        "reason": (
                            "The audit encountered a candidate-specific processing "
                            "error; inspect the recorded error before relying on it."
                        ),
                        "issues": ["CANDIDATE_AUDIT_ERROR"],
                        "original_question": candidate.get("question", ""),
                        "original_answer": candidate.get("answer", ""),
                        "original_reference": candidate.get("reference", ""),
                        "source_document": candidate.get("source_document", ""),
                        "declared_pdf_page": candidate.get("page", ""),
                        "supporting_passage_supplied": candidate.get(
                            "supporting_passage", ""
                        ),
                        "exact_source_evidence_excerpt": "",
                        "matched_source_pdf_page": None,
                        "reference_check_status": "NOT_CHECKED",
                        "reference_check_reason": "Audit processing error.",
                        "answer_exactly_supported_by_supplied_passage": False,
                        "suggested_question": "",
                        "suggested_answer": "",
                        "suggested_reference": "",
                        "suggestion_status": "NO_AUTOMATIC_CORRECTION",
                        "existing_suggestion": {},
                        "prior_ai_recommendations": candidate.get(
                            "prior_ai_recommendations", []
                        ),
                        "prior_decisions_not_used_as_validation": candidate.get(
                            "prior_decisions", []
                        ),
                        "source_occurrences": candidate.get("source_occurrences", []),
                        "source_record_count": candidate.get("source_record_count", 1),
                        "source_field_variants": candidate.get(
                            "source_field_variants", {}
                        ),
                        "duplicate_group_id": "",
                        "duplicate_candidate_ids": [],
                        "duplicate_relation": "",
                        "preferred_duplicate_candidate_id": "",
                        "preferred_candidate_reason": "",
                        "audit_error": f"{type(error).__name__}: {error}",
                        "pdf_read_error": "",
                        "current_law_status": "NOT_CHECKED",
                    })
            with checkpoint_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(batch_results, stream, ensure_ascii=False)
                stream.write("\n")
        for result in batch_results:
            audited_by_id[str(result["candidate_id"])] = result
        print(
            f"Batch {batch_index + 1}/{total_batches}: audited "
            f"{min(start + len(batch_candidates), len(candidates))}/"
            f"{len(candidates)} candidate identities."
        )

    records = [audited_by_id[candidate["candidate_id"]] for candidate in candidates]
    duplicate_groups = detect_duplicate_groups(records)
    candidate_errors = sum(
        "CANDIDATE_AUDIT_ERROR" in record.get("issues", [])
        for record in records
    )
    summary = {
        "generated_at_utc": _now(),
        "input_file_count": len(inputs),
        "source_occurrences_loaded": len(input_occurrences),
        "input_error_count": len(input_errors),
        "input_errors": input_errors,
        "candidate_error_count": candidate_errors,
        "source_pdf_errors": pdf_errors,
        "source_pdf_paths": pdf_paths,
        "source_pdfs_checked": list(SOURCE_PDFS),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "batch_size": batch_size,
        "input_fingerprint": fingerprint,
        "historical_or_ai_labels_used_as_approval": False,
        "human_approval_status": "NOT_SET_BY_AUTOMATION",
        "legal_correctness_status": "NOT_VERIFIED",
        "output_directory": str(output_dir.resolve()),
    }
    write_final_reports(output_dir, records, duplicate_groups, summary)
    manifest["completed"] = True
    manifest["completed_at_utc"] = _now()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Full audit reports written to: {output_dir}")
    return output_dir
