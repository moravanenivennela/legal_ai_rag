"""Build source-grounded QA candidates without loading a language model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDFS = {
    "constitution_of_india.pdf": ROOT / "data" / "constitution_of_india.pdf",
    "consumer_protection_act_2019.pdf": (
        ROOT / "data" / "consumer_protection_act_2019.pdf"
    ),
}
PILOT_OUTPUT_DIR = ROOT / "reports" / "qa_generation_pilot_20261004_v2"
SEED = 42
NEAR_DUPLICATE_THRESHOLD = 0.90
SCHEMA_VERSION = "1.0"
HUMAN_REVIEW_STATUS = "PENDING_HUMAN_REVIEW"
DEFAULT_CANDIDATE_LIMIT = 1000
STAGE12_OUTPUT_DIR = (
    ROOT / "reports" / "qa_candidate_generation_20261004T084249Z_seed42"
)
REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "passage_id",
    "question",
    "answer",
    "evidence",
    "source_filename",
    "page",
    "legal_reference",
    "reference_status",
    "question_type",
    "review_status",
)
LEGAL_REFERENCE_RE = re.compile(
    r"^(?:Article|Section)\s+\d+[A-Za-z]?(?:\(\d+[A-Za-z]?\))*$",
    re.IGNORECASE,
)
ARTICLE_HEADING_RE = re.compile(
    r"^\s*(\d+[A-Za-z]?)\.\s+([^.\n]{2,100})\.\s*",
    re.IGNORECASE,
)
ARTICLE_SEGMENT_RE = re.compile(
    r"(?m)^\s*(\d+[A-Za-z]?)\.\s+([^.\n]{2,100})\.\s*[—–-]\s*"
)
CONSUMER_SECTION_SEGMENT_RE = re.compile(
    r"(?m)^\s*(\d+[A-Za-z]?)\.\s+([^.\n]{2,120})\.\s*[—–-]\s*"
)
SECTION_HEADING_RE = re.compile(
    r"\b(\d+)\.\s+In this Act,\s+unless the context otherwise requires",
    re.IGNORECASE,
)
DEFINITION_RE = re.compile(
    r"""(?is)(?:\(\s*(\d+)\s*\)\s*)?["“]?([A-Za-z][A-Za-z '-]{1,60})["”]?\s+means\b(?!\s+of\b)"""
)
SUBITEM_RE = re.compile(r"\(\s*(\d+)\s*\)")
CONTENTS_RE = re.compile(r"\bcontents\b", re.IGNORECASE)
ANSWER_TYPE_PRIORITY = {
    "definition": 0,
    "condition_exception": 1,
    "procedure_remedy": 2,
    "direct_factual": 3,
    "rights_duties": 4,
    "scenario": 5,
    "comparison": 6,
}
QUESTION_TYPES = frozenset(ANSWER_TYPE_PRIORITY)


def normalize_text(value: Any) -> str:
    """Normalize extracted text for matching while retaining source text."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def passage_id(source_filename: str, page: int, evidence: str) -> str:
    digest = hashlib.sha256(
        f"{source_filename}\0{page}\0{normalize_text(evidence)}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{Path(source_filename).stem}-{page}-{digest}"


def extract_pdf_passages(
    pdf_path: Path,
    source_filename: str | None = None,
) -> list[dict[str, Any]]:
    """Extract one exact, page-numbered text passage from each nonempty PDF page."""
    source = source_filename or pdf_path.name
    passages = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            evidence = text.strip()
            if not evidence:
                continue
            passages.append({
                "passage_id": passage_id(source, page_number, evidence),
                "source_filename": source,
                "page": page_number,
                "evidence": evidence,
            })
    return passages


def extract_selected_pdf_pages(
    pdf_path: Path,
    page_numbers: Sequence[int],
) -> dict[int, str]:
    """Extract only requested 1-based PDF pages, preserving the page numbering."""
    with pymupdf.open(pdf_path) as document:
        result = {}
        for page_number in page_numbers:
            if page_number < 1 or page_number > len(document):
                raise ValueError(
                    f"Requested PDF page {page_number} is outside {pdf_path.name}."
                )
            result[page_number] = document[page_number - 1].get_text(
                "text",
                sort=True,
            )
        return result


def extract_pilot_passages(
    source_pages: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Select the same two small, explicit provision windows used by the pilot."""
    constitution = _page(source_pages, "constitution_of_india.pdf", 37)
    consumer = _page(source_pages, "consumer_protection_act_2019.pdf", 2)

    article_start = re.search(
        r"(?im)^\s*14\.\s*Equality before law\.",
        constitution,
    )
    article_end = re.search(
        r"(?im)^\s*15\.\s*Prohibition of discrimination",
        constitution,
    )
    if not article_start or not article_end or article_end.start() <= article_start.start():
        raise ValueError("Could not isolate the Article 14 source passage on PDF page 37.")
    article_text = constitution[article_start.start():article_end.start()].strip()

    section_start = re.search(
        r"(?i)\b2\.\s*In this Act,\s*unless the context otherwise requires",
        consumer,
    )
    definition_start = re.search(
        r"""(?i)\(\s*1\s*\)\s*["“]?advertisement["”]?\s+means""",
        consumer,
    )
    definition_end = re.search(
        r"(?i)\(\s*2\s*\)",
        consumer,
    )
    if (
        not section_start
        or not definition_start
        or not definition_end
        or section_start.start() > definition_start.start()
        or definition_end.start() <= definition_start.start()
    ):
        raise ValueError(
            "Could not isolate the Section 2(1) definition on PDF page 2."
        )
    consumer_text = consumer[section_start.start():definition_end.start()].strip()

    return [
        _make_passage("constitution_of_india.pdf", 37, article_text),
        _make_passage("consumer_protection_act_2019.pdf", 2, consumer_text),
    ]


def split_page_into_passages(
    source_filename: str,
    page: int,
    page_text: str,
) -> list[dict[str, Any]]:
    """Split a PDF page only at explicit provision-heading typography."""
    if CONTENTS_RE.search(page_text):
        return [_make_passage(source_filename, page, page_text.strip())]
    if Path(source_filename).name == "constitution_of_india.pdf":
        heading_pattern = ARTICLE_SEGMENT_RE
    elif Path(source_filename).name == "consumer_protection_act_2019.pdf":
        headings = sorted(
            list(CONSUMER_SECTION_SEGMENT_RE.finditer(page_text))
            + list(SECTION_HEADING_RE.finditer(page_text)),
            key=lambda match: match.start(),
        )
        headings = [
            match for index, match in enumerate(headings)
            if index == 0 or match.start() != headings[index - 1].start()
        ]
        if not headings:
            return [_make_passage(source_filename, page, page_text.strip())]
        passages = []
        for index, heading in enumerate(headings):
            end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(page_text)
            )
            evidence = page_text[heading.start():end].strip()
            if evidence:
                passages.append(_make_passage(source_filename, page, evidence))
        return passages or [_make_passage(source_filename, page, page_text.strip())]
    else:
        return [_make_passage(source_filename, page, page_text.strip())]

    headings = list(heading_pattern.finditer(page_text))
    if not headings:
        return [_make_passage(source_filename, page, page_text.strip())]

    passages = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(page_text)
        evidence = page_text[heading.start():end].strip()
        if evidence:
            passages.append(_make_passage(source_filename, page, evidence))
    return passages or [_make_passage(source_filename, page, page_text.strip())]


def _page(
    source_pages: Mapping[str, Sequence[str]],
    source_filename: str,
    page_number: int,
) -> str:
    pages = source_pages.get(source_filename)
    if pages is None or len(pages) < page_number:
        raise ValueError(
            f"Source {source_filename} does not contain PDF page {page_number}."
        )
    return pages[page_number - 1]


def _make_passage(source_filename: str, page: int, evidence: str) -> dict[str, Any]:
    reference, reference_status = extract_legal_reference(source_filename, evidence)
    return {
        "passage_id": passage_id(source_filename, page, evidence),
        "source_filename": source_filename,
        "page": page,
        "legal_reference": reference,
        "reference_status": reference_status,
        "evidence": evidence,
    }


def extract_legal_reference(
    source_filename: str,
    evidence: str,
) -> tuple[str | None, str]:
    """Recognize only a provision number explicitly present in its source context."""
    if CONTENTS_RE.search(evidence):
        return None, "UNCONFIRMED"

    if Path(source_filename).name == "constitution_of_india.pdf":
        heading = ARTICLE_HEADING_RE.match(evidence)
        if heading:
            return f"Article {heading.group(1)}", "EXPLICIT_NUMBERED_HEADING"
        return None, "UNCONFIRMED"

    if Path(source_filename).name == "consumer_protection_act_2019.pdf":
        section = SECTION_HEADING_RE.search(evidence)
        if section:
            section_number = section.group(1)
        else:
            section_heading = CONSUMER_SECTION_SEGMENT_RE.match(evidence)
            if not section_heading:
                return None, "UNCONFIRMED"
            section_number = section_heading.group(1)
        if not section_number:
            return None, "UNCONFIRMED"
        definition = DEFINITION_RE.search(evidence)
        reference = f"Section {section_number}"
        if definition and definition.group(1):
            reference += f"({definition.group(1)})"
        return reference, "EXPLICIT_NUMBERED_HEADING"

    return None, "UNCONFIRMED"


def _answer_fingerprint(answer: str) -> str:
    """Collapse exact answer templates after replacing explicit legal numbers."""
    text = normalize_text(answer)
    text = re.sub(r"\b(?:article|section)\s+\d+[a-z]?(?:\s+\d+)*\b", "<ref>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    return text


def generate_passage_candidates(
    passage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create extractive candidates only from recognized, explicit source patterns."""
    evidence = str(passage.get("evidence", ""))
    source_filename = str(passage.get("source_filename", ""))
    legal_reference = passage.get("legal_reference")
    raw: list[dict[str, Any]] = []

    definition = DEFINITION_RE.search(evidence)
    if definition:
        definitions = list(DEFINITION_RE.finditer(evidence))
        for index, definition_match in enumerate(definitions):
            term = definition_match.group(2).strip()
            end = (
                definitions[index + 1].start()
                if index + 1 < len(definitions)
                else len(evidence)
            )
            clause_end = evidence.find(";", definition_match.start(), end)
            if clause_end >= 0:
                end = clause_end + 1
            answer = evidence[definition_match.start():end].strip()
            definition_reference = legal_reference
            if (
                definition_reference
                and definition_match.group(1)
                and definition_reference.startswith("Section ")
            ):
                section_number = definition_reference.split("(", 1)[0]
                definition_reference = (
                    f"{section_number}({definition_match.group(1)})"
                )
            reference = definition_reference or "the Act"
            raw.append(_make_candidate(
                passage,
                "definition",
                f"How does {reference} define {term}?",
                answer,
                legal_reference=definition_reference,
                candidate_discriminator=f"definition:{definition_match.start()}",
            ))

    article_heading = (
        ARTICLE_HEADING_RE.match(evidence)
        if Path(source_filename).name == "constitution_of_india.pdf"
        else None
    )
    if article_heading:
        body_start = article_heading.end()
        body = re.sub(r"^\s*[—–-]\s*", "", evidence[body_start:]).strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0].strip()
        heading_text = article_heading.group(2).strip().rstrip("— ")
        if first_sentence and re.search(r"\b(?:shall|may|is|are|means)\b", first_sentence, re.I):
            raw.append(_make_candidate(
                passage,
                "direct_factual",
                f"What does {legal_reference or 'this provision'} state about {heading_text}?",
                first_sentence,
            ))
        duty = re.search(
            r"\b(The\s+[^.;]{1,100}?\s+shall\s+not\s+[^.;]+[.;])",
            first_sentence,
            re.IGNORECASE,
        )
        if duty:
            raw.append(_make_candidate(
                passage,
                "rights_duties",
                f"What does the provision state the State shall not do?",
                duty.group(1).strip(),
            ))

    for match in re.finditer(r"(?is)(?:provided that|unless|if)\b[^.;]*[.;]?", evidence):
        clause = match.group(0).strip()
        normalized_clause = normalize_text(clause)
        if (
            len(normalized_clause.split()) >= 5
            and "context otherwise requires" not in normalized_clause
        ):
            raw.append(_make_candidate(
                passage,
                "condition_exception",
                (
                    "What condition or exception is described in the clause "
                    f"beginning “{_question_anchor(clause)}”?"
                ),
                clause,
                candidate_discriminator=f"condition:{match.start()}",
            ))

    for match in re.finditer(
        r"(?is)\bif\s+(?P<condition>[^,;]{3,180})[,;]\s*"
        r"(?P<result>[^.;]{3,260})[.;]?",
        evidence,
    ):
        clause = match.group(0).strip()
        raw.append(_make_candidate(
            passage,
            "scenario",
            f"According to the passage, what happens if {match.group('condition').strip()}?",
            clause,
            candidate_discriminator=f"scenario:{match.start()}",
        ))

    for match in re.finditer(r"(?is)[^.;]*(?:appeal|file a complaint|redress|remedy)[^.;]*[.;]?", evidence):
        clause = match.group(0).strip()
        if len(normalize_text(clause).split()) >= 5:
            raw.append(_make_candidate(
                passage,
                "procedure_remedy",
                    (
                        "What procedure or remedy is described in the clause "
                        f"beginning “{_question_anchor(clause)}”?"
                    ),
                    clause,
                    candidate_discriminator=f"procedure:{match.start()}",
                ))

    for match in re.finditer(r"(?is)[^.;]*(?:right to|shall|must|is entitled to)[^.;]*[.;]?", evidence):
        clause = match.group(0).strip()
        if len(normalize_text(clause).split()) >= 5:
            if not any(
                normalize_text(item["answer"]) == normalize_text(clause)
                and item["question_type"] == "rights_duties"
                for item in raw
            ):
                raw.append(_make_candidate(
                    passage,
                    "rights_duties",
                    (
                        "What right, duty, or requirement is described in the "
                        f"clause beginning “{_question_anchor(clause)}”?"
                    ),
                    clause,
                    candidate_discriminator=f"rights:{match.start()}",
                ))

    comparison = re.search(
        r"(?is)[^.;]*(?:whereas|as compared with|in contrast to)[^.;]*[.;]?",
        evidence,
    )
    if comparison:
        clause = comparison.group(0).strip()
        raw.append(_make_candidate(
            passage,
            "comparison",
            "What comparison does the passage explicitly make?",
            clause,
            candidate_discriminator=f"comparison:{comparison.start()}",
        ))

    return raw


def _question_anchor(clause: str, max_words: int = 8) -> str:
    """Use a short literal clause prefix to distinguish otherwise generic prompts."""
    words = re.findall(r"\S+", re.sub(r"\s+", " ", clause).strip())
    anchor = " ".join(words[:max_words]).strip(" ,;:")
    if len(words) > max_words:
        anchor += "…"
    return anchor


def _make_candidate(
    passage: Mapping[str, Any],
    question_type: str,
    question: str,
    answer: str,
    legal_reference: str | None = None,
    candidate_discriminator: str = "",
) -> dict[str, Any]:
    question = re.sub(r"\s+", " ", question).strip()
    answer = answer.strip()
    candidate_key = "\0".join((
        str(passage.get("passage_id", "")),
        question_type,
        normalize_text(question),
        normalize_text(answer),
        candidate_discriminator,
    ))
    candidate_id = hashlib.sha256(candidate_key.encode("utf-8")).hexdigest()[:16]
    return {
        "candidate_id": candidate_id,
        "passage_id": passage.get("passage_id"),
        "question": question,
        "answer": answer,
        "evidence": passage.get("evidence", ""),
        "source_filename": passage.get("source_filename", ""),
        "page": passage.get("page"),
        "legal_reference": (
            legal_reference
            if legal_reference is not None
            else passage.get("legal_reference")
        ),
        "reference_status": (
            "EXPLICIT_NUMBERED_HEADING"
            if legal_reference is not None
            else passage.get("reference_status", "UNCONFIRMED")
        ),
        "question_type": question_type,
        "review_status": HUMAN_REVIEW_STATUS,
    }


def deduplicate_candidates(
    candidates: Iterable[dict[str, Any]],
    near_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Remove duplicate questions; flag answer repetition without dropping distinct Qs."""
    ordered = sorted(
        candidates,
        key=lambda item: (
            ANSWER_TYPE_PRIORITY.get(str(item.get("question_type")), 99),
            str(item.get("source_filename", "")),
            int(item.get("page") or 0),
            normalize_text(item.get("question")),
        ),
    )
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    question_groups: dict[str, list[dict[str, Any]]] = {}
    answer_groups: dict[str, list[dict[str, Any]]] = {}
    template_groups: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        template = _answer_fingerprint(str(row.get("answer", "")))
        if question:
            question_groups.setdefault(question, []).append(row)
        if answer:
            answer_groups.setdefault(answer, []).append(row)
        if template:
            template_groups.setdefault(template, []).append(row)

    for duplicate_rows in answer_groups.values():
        if len(duplicate_rows) > 1:
            for row in duplicate_rows:
                flags = row.setdefault("duplicate_flags", [])
                if "REPEATED_EXACT_ANSWER" not in flags:
                    flags.append("REPEATED_EXACT_ANSWER")
    for duplicate_rows in template_groups.values():
        if len(duplicate_rows) > 1:
            for row in duplicate_rows:
                flags = row.setdefault("duplicate_flags", [])
                if "REPEATED_ANSWER_TEMPLATE" not in flags:
                    flags.append("REPEATED_ANSWER_TEMPLATE")

    for candidate in ordered:
        question = normalize_text(candidate.get("question"))
        duplicate_of = None
        reason = ""
        for prior in kept:
            prior_question = normalize_text(prior.get("question"))
            if question and question == prior_question:
                duplicate_of = str(prior["candidate_id"])
                reason = "EXACT_QUESTION_DUPLICATE"
                break
            if not question or not prior_question:
                continue
            left_answer = _answer_fingerprint(str(candidate.get("answer", "")))
            right_answer = _answer_fingerprint(str(prior.get("answer", "")))
            if (
                candidate.get("question_type") == prior.get("question_type")
                and left_answer
                and left_answer == right_answer
                and abs(len(question) - len(prior_question))
                / max(len(question), len(prior_question))
                <= 1 - near_threshold
                and SequenceMatcher(None, question, prior_question).ratio()
                >= near_threshold
            ):
                duplicate_of = str(prior["candidate_id"])
                reason = "NEAR_QUESTION_DUPLICATE"
                break
        if duplicate_of:
            flags = candidate.setdefault("duplicate_flags", [])
            if reason not in flags:
                flags.append(reason)
            rejected.append({
                "candidate_id": str(candidate.get("candidate_id", "")),
                "duplicate_of": duplicate_of,
                "reason": reason,
            })
            continue
        kept.append(candidate)
    return kept, rejected


def select_source_balanced_candidates(
    candidates: Sequence[dict[str, Any]],
    per_source_limit: int,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Apply an equal maximum per source; never fill a short source synthetically."""
    if per_source_limit < 1:
        raise ValueError("per_source_limit must be positive.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["source_filename"]), []).append(candidate)
    rng = random.Random(seed)
    selected = []
    for source_filename in sorted(grouped):
        rows = sorted(grouped[source_filename], key=lambda row: str(row["candidate_id"]))
        rng.shuffle(rows)
        selected.extend(rows[:per_source_limit])
    return selected


def select_balanced_candidates(
    candidates: Sequence[dict[str, Any]],
    candidate_limit: int,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Retain the whole pool under the limit; otherwise sample proportionally."""
    if candidate_limit < 0:
        raise ValueError("candidate_limit must be zero or greater.")
    if candidate_limit == 0:
        return []
    if len(candidates) <= candidate_limit:
        return sorted(
            candidates,
            key=lambda row: (
                str(row["source_filename"]),
                str(row["question_type"]),
                str(row["candidate_id"]),
            ),
        )
    rng = random.Random(seed)
    rows = list(candidates)
    source_availability = Counter(str(row["source_filename"]) for row in rows)
    type_availability = Counter(str(row["question_type"]) for row in rows)
    rng.shuffle(rows)
    tie_order = {
        str(row["candidate_id"]): index
        for index, row in enumerate(rows)
    }
    source_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    selected = []
    remaining = {str(row["candidate_id"]): row for row in rows}
    if candidate_limit >= len(source_availability):
        for source_filename in sorted(source_availability):
            representative = next(
                row for row in rows
                if str(row["source_filename"]) == source_filename
            )
            selected.append(representative)
            source_counts[source_filename] += 1
            type_counts[str(representative["question_type"])] += 1
            remaining.pop(str(representative["candidate_id"]))

    while remaining and len(selected) < candidate_limit:
        selected_target = len(selected) + 1
        next_candidate = max(
            remaining.values(),
            key=lambda row: (
                selected_target
                * source_availability[str(row["source_filename"])]
                / len(rows)
                - source_counts[str(row["source_filename"])],
                selected_target
                * type_availability[str(row["question_type"])]
                / len(rows)
                - type_counts[str(row["question_type"])],
                -tie_order[str(row["candidate_id"])],
            ),
        )
        selected.append(next_candidate)
        source_counts[str(next_candidate["source_filename"])] += 1
        type_counts[str(next_candidate["question_type"])] += 1
        remaining.pop(str(next_candidate["candidate_id"]))
    return selected


def validate_candidate(
    candidate: Mapping[str, Any],
    source_page_texts: Mapping[tuple[str, int], str],
) -> list[str]:
    errors = [
        f"Missing required field: {field}"
        for field in REQUIRED_CANDIDATE_FIELDS
        if field not in candidate
    ]
    if errors:
        return errors

    for field in ("question", "answer", "evidence", "source_filename", "passage_id"):
        if not str(candidate.get(field) or "").strip():
            errors.append(f"Empty required field: {field}")

    page = candidate.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        errors.append("Page must be a positive 1-based integer.")
    else:
        page_text = source_page_texts.get((str(candidate["source_filename"]), page))
        if page_text is None:
            errors.append("Source/page pair is not present in extracted source text.")
        elif normalize_text(candidate["evidence"]) not in normalize_text(page_text):
            errors.append("Evidence does not match the declared source page.")

    legal_reference = candidate.get("legal_reference")
    if legal_reference is not None:
        if not LEGAL_REFERENCE_RE.fullmatch(str(legal_reference)):
            errors.append("Legal reference does not match the supported format.")
        elif candidate.get("reference_status") != "EXPLICIT_NUMBERED_HEADING":
            errors.append("Legal reference is not marked as explicitly identified.")
        elif not reference_supported(
            str(candidate["source_filename"]),
            str(candidate["evidence"]),
            str(legal_reference),
        ):
            errors.append("Legal reference is not supported by the supplied passage.")
    elif candidate.get("reference_status") != "UNCONFIRMED":
        errors.append("Missing legal reference must remain UNCONFIRMED.")

    if candidate.get("reference_status") not in (
        "EXPLICIT_NUMBERED_HEADING",
        "UNCONFIRMED",
    ):
        errors.append("Reference status is not a supported deterministic value.")

    if (
        normalize_text(candidate.get("answer"))
        and normalize_text(candidate.get("answer"))
        not in normalize_text(candidate.get("evidence"))
    ):
        errors.append("Answer is not an extract from the linked evidence.")

    if candidate.get("review_status") != HUMAN_REVIEW_STATUS:
        errors.append("Candidate must remain pending human legal review.")
    if not str(candidate.get("question_type", "")).strip():
        errors.append("Question type is empty.")
    elif candidate.get("question_type") not in QUESTION_TYPES:
        errors.append("Question type is not supported by this pipeline.")
    if candidate.get("split") not in (None, "train", "validation", "test"):
        errors.append("Split must be train, validation, test, or unset.")
    return errors


def reference_supported(
    source_filename: str,
    evidence: str,
    legal_reference: str,
) -> bool:
    """Check a typed reference against the explicit heading in its evidence."""
    if Path(source_filename).name == "constitution_of_india.pdf":
        match = ARTICLE_HEADING_RE.match(evidence)
        return bool(match and legal_reference.casefold() == f"article {match.group(1)}".casefold())
    if Path(source_filename).name != "consumer_protection_act_2019.pdf":
        return False

    section = SECTION_HEADING_RE.search(evidence)
    if section:
        section_number = section.group(1)
    else:
        heading = CONSUMER_SECTION_SEGMENT_RE.match(evidence)
        if not heading:
            return False
        section_number = heading.group(1)

    expected = f"Section {section_number}"
    subitem = re.fullmatch(
        r"Section\s+(\d+[A-Za-z]?)\((\d+[A-Za-z]?)\)",
        legal_reference,
        re.IGNORECASE,
    )
    if subitem:
        if subitem.group(1).casefold() != section_number.casefold():
            return False
        return any(
            match.group(1)
            and match.group(1).casefold() == subitem.group(2).casefold()
            for match in DEFINITION_RE.finditer(evidence)
        )
    return legal_reference.casefold() == expected.casefold()


def validate_split_leakage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    near_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[str]:
    errors = []
    split_rows = [
        (split_name, candidate)
        for split_name, rows in splits.items()
        for candidate in rows
    ]
    for index, (left_split, left) in enumerate(split_rows):
        for right_split, right in split_rows[index + 1:]:
            if left_split == right_split:
                continue
            if left.get("passage_id") and left.get("passage_id") == right.get("passage_id"):
                errors.append(
                    f"Passage leakage across {left_split}/{right_split}: "
                    f"{left.get('passage_id')}"
                )
            left_question = normalize_text(left.get("question"))
            right_question = normalize_text(right.get("question"))
            if left_question and right_question and (
                left_question == right_question
                or SequenceMatcher(None, left_question, right_question).ratio()
                >= near_threshold
            ):
                errors.append(
                    f"Question leakage across {left_split}/{right_split}: "
                    f"{left.get('candidate_id')} / {right.get('candidate_id')}"
                )
            if _answer_fingerprint(str(left.get("answer", ""))) and (
                _answer_fingerprint(str(left.get("answer", "")))
                == _answer_fingerprint(str(right.get("answer", "")))
            ):
                errors.append(
                    f"Answer-template leakage across {left_split}/{right_split}: "
                    f"{left.get('candidate_id')} / {right.get('candidate_id')}"
                )
    return errors


def build_human_review_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "candidate_id": candidate["candidate_id"],
        "question": candidate["question"],
        "answer": candidate["answer"],
        "source_filename": candidate["source_filename"],
        "page": candidate["page"],
        "legal_reference": candidate["legal_reference"] or "",
        "evidence": candidate["evidence"],
        "question_type": candidate["question_type"],
        "validation_flags": "; ".join(candidate.get("validation_flags", [])),
        "duplicate_flags": "; ".join(candidate.get("duplicate_flags", [])),
        "selection_flags": "; ".join(candidate.get("selection_flags", [])),
        "review_status": HUMAN_REVIEW_STATUS,
        "reviewer_decision": "",
        "reviewer_notes": "",
    } for candidate in candidates]


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else [
        "passage_id",
        "source_filename",
        "page",
        "legal_reference",
        "reason",
        "evidence",
    ]
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _split_flags(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(flag).strip() for flag in value if str(flag).strip()]
    return [flag.strip() for flag in str(value or "").split(";") if flag.strip()]


def _distribution(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    missing_label: str = "UNCONFIRMED",
) -> dict[str, int]:
    return dict(Counter(
        str(row.get(field) or missing_label)
        for row in rows
    ))


def _flag_distribution(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(Counter(
        flag
        for row in rows
        for flag in _split_flags(row.get(field))
    ))


def _load_stage12_comparison() -> dict[str, Any]:
    if not STAGE12_OUTPUT_DIR.is_dir():
        return {"available": False, "reason": "Stage 12 directory not found."}

    with (STAGE12_OUTPUT_DIR / "human_review_report.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        old_raw = list(csv.DictReader(stream))
    old_manifest = json.loads(
        (STAGE12_OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8")
    )
    old_selected = [
        json.loads(line)
        for line in (STAGE12_OUTPUT_DIR / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    def reference_status(row: Mapping[str, Any]) -> str:
        reference = str(row.get("legal_reference") or "")
        if not reference:
            return "UNCONFIRMED"
        if LEGAL_REFERENCE_RE.fullmatch(reference):
            return "REFERENCE_PRESENT_NOT_LEGAL_VERIFIED"
        return "MALFORMED_REFERENCE"

    return {
        "available": True,
        "stage12_directory": str(STAGE12_OUTPUT_DIR.relative_to(ROOT)),
        "stage12": {
            "raw_count": len(old_raw),
            "distinct_question_count": len({
                normalize_text(row.get("question"))
                for row in old_raw
                if normalize_text(row.get("question"))
            }),
            "deduplicated_pool_count": int(
                old_manifest["counts"]["retained_candidates"]
                + old_manifest["counts"]["candidates_omitted_by_selection_caps"]
            ),
            "selected_count": len(old_selected),
            "raw_by_source": _distribution(old_raw, "source_filename"),
            "deduplicated_by_source": old_manifest["settings"][
                "candidate_pool_distribution"
            ]["deduplicated_by_source"],
            "selected_by_source": _distribution(old_selected, "source_filename"),
            "raw_by_question_type": _distribution(old_raw, "question_type"),
            "deduplicated_by_question_type": old_manifest["settings"][
                "candidate_pool_distribution"
            ]["deduplicated_by_question_type"],
            "selected_by_question_type": _distribution(
                old_selected,
                "question_type",
            ),
            "reference_status": dict(Counter(reference_status(row) for row in old_raw)),
            "duplicate_category": _flag_distribution(old_raw, "duplicate_flags"),
            "selection_reason": _flag_distribution(old_raw, "selection_flags"),
        },
    }


def _compare_stage12(
    generated: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = _load_stage12_comparison()
    if not comparison.get("available"):
        return comparison

    raw = generated["raw_candidates"]
    deduplicated = generated["deduplicated"]
    selected = generated["selected"]
    comparison["stage13"] = {
        "raw_count": len(raw),
        "raw_distinct_question_count": len({
            normalize_text(row.get("question")) for row in raw
            if normalize_text(row.get("question"))
        }),
        "deduplicated_pool_count": len(deduplicated),
        "deduplicated_distinct_question_count": len({
            normalize_text(row.get("question")) for row in deduplicated
            if normalize_text(row.get("question"))
        }),
        "selected_count": len(selected),
        "selected_distinct_question_count": len({
            normalize_text(row.get("question")) for row in selected
            if normalize_text(row.get("question"))
        }),
        "raw_by_source": _distribution(raw, "source_filename"),
        "deduplicated_by_source": _distribution(deduplicated, "source_filename"),
        "selected_by_source": _distribution(selected, "source_filename"),
        "raw_by_question_type": _distribution(raw, "question_type"),
        "deduplicated_by_question_type": _distribution(
            deduplicated,
            "question_type",
        ),
        "selected_by_question_type": _distribution(selected, "question_type"),
        "reference_status": dict(Counter(
            str(row.get("reference_status") or "UNCONFIRMED")
            for row in raw
        )),
        "duplicate_category": _flag_distribution(raw, "duplicate_flags"),
        "selection_reason": _flag_distribution(raw, "selection_flags"),
    }
    return comparison


def write_run_outputs(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    failures: Sequence[str],
    settings: Mapping[str, Any],
    duplicate_rejections: Sequence[Mapping[str, str]] = (),
    review_candidates: Sequence[Mapping[str, Any]] | None = None,
    no_candidate_passages: Sequence[Mapping[str, Any]] = (),
    comparison_to_stage12: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a separate candidate file, reviewer worksheet, and run manifest."""
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = output_dir / "candidates.jsonl"
    review_path = output_dir / "human_review_report.csv"
    no_candidate_path = output_dir / "no_candidate_passages.csv"
    manifest_path = output_dir / "run_manifest.json"
    _write_jsonl_exclusive(candidate_path, candidates)
    review_rows = review_candidates if review_candidates is not None else candidates
    _write_csv_exclusive(review_path, build_human_review_rows(review_rows))
    _write_csv_exclusive(no_candidate_path, no_candidate_passages)

    source_counts = Counter(str(row["source_filename"]) for row in candidates)
    reference_counts = Counter(
        str(row["legal_reference"] or "UNCONFIRMED") for row in candidates
    )
    type_counts = Counter(str(row["question_type"]) for row in candidates)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic extractive templates; no language model used",
        "model": None,
        "ollama_used": False,
        "settings": dict(settings),
        "counts": {
            "raw_candidates": settings.get(
                "raw_candidate_count",
                len(candidates) + len(duplicate_rejections),
            ),
            "invalid_candidates": settings.get("invalid_candidate_count", 0),
            "retained_candidates": len(candidates),
            "duplicate_candidates_removed": len(duplicate_rejections),
            "candidates_omitted_by_candidate_limit": settings.get(
                "omitted_by_candidate_limit",
                settings.get("omitted_by_selection_caps", 0),
            ),
            "failures": len(failures),
            "human_review_pending": len(review_rows),
        },
        "duplicate_rejections": list(duplicate_rejections),
        "failures": list(failures),
        "coverage": {
            "source_document": dict(source_counts),
            "legal_reference": dict(reference_counts),
            "question_type": dict(type_counts),
        },
        "candidate_quality_flags": {
            "duplicate_categories": _flag_distribution(
                review_rows,
                "duplicate_flags",
            ),
            "selection_reasons": _flag_distribution(
                review_rows,
                "selection_flags",
            ),
            "validation_flags": _flag_distribution(
                review_rows,
                "validation_flags",
            ),
            "reference_status": dict(Counter(
                str(row.get("reference_status") or "UNCONFIRMED")
                for row in review_rows
            )),
        },
        "no_candidate_passages": {
            "count": len(no_candidate_passages),
            "reason_counts": _distribution(
                no_candidate_passages,
                "reason",
                "UNCLASSIFIED",
            ),
        },
        "comparison_to_stage12": dict(comparison_to_stage12 or {}),
        "review_status_counts": {
            HUMAN_REVIEW_STATUS: len(review_rows),
        },
        "outputs": {
            "candidates": candidate_path.name,
            "human_review_report": review_path.name,
            "no_candidate_passages": no_candidate_path.name,
            "manifest": manifest_path.name,
        },
        "legal_correctness_claim": (
            "None. Automated provenance checks and templates do not establish "
            "legal correctness; every row requires human review."
        ),
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def _run_pilot(output_dir: Path) -> dict[str, Any]:
    source_pages: dict[str, list[str]] = {}
    source_page_texts: dict[tuple[str, int], str] = {}
    failures = []
    pilot_pages = {
        "constitution_of_india.pdf": 37,
        "consumer_protection_act_2019.pdf": 2,
    }
    for filename, pdf_path in SOURCE_PDFS.items():
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Required source PDF is missing: {pdf_path}")
        page_number = pilot_pages[filename]
        page_text = extract_selected_pdf_pages(pdf_path, [page_number])[page_number]
        page_list = [""] * page_number
        page_list[page_number - 1] = page_text
        source_pages[filename] = page_list
        source_page_texts[(filename, page_number)] = page_text
    selected = extract_pilot_passages(source_pages)

    raw_candidates = []
    for passage in selected:
        generated = generate_passage_candidates(passage)
        if not generated:
            failures.append(
                f"No supported template matched passage {passage['passage_id']}."
            )
        raw_candidates.extend(generated)

    validated = []
    for candidate in raw_candidates:
        errors = validate_candidate(candidate, source_page_texts)
        if errors:
            failures.append(
                f"{candidate['candidate_id']}: {'; '.join(errors)}"
            )
        else:
            validated.append(candidate)

    deduplicated, duplicate_rejections = deduplicate_candidates(validated)
    settings = {
        "seed": SEED,
        "random_sampling": False,
        "passages": [
            {"source_filename": row["source_filename"], "page": row["page"]}
            for row in selected
        ],
        "max_raw_candidates": 4,
        "near_duplicate_question_threshold": NEAR_DUPLICATE_THRESHOLD,
        "reference_policy": (
            "Only explicit numbered provision headings; otherwise null and UNCONFIRMED."
        ),
    }
    manifest = write_run_outputs(
        output_dir,
        deduplicated,
        failures,
        settings,
        duplicate_rejections,
    )
    return manifest


def generate_from_passages(
    passages: Sequence[Mapping[str, Any]],
    source_page_texts: Mapping[tuple[str, int], str],
    candidate_limit: int,
    seed: int,
) -> dict[str, Any]:
    raw_candidates = []
    failures = []
    no_candidate_passages = []
    for passage in passages:
        generated = generate_passage_candidates(passage)
        if not generated:
            evidence = str(passage.get("evidence", ""))
            if CONTENTS_RE.search(evidence):
                reason = "CONTENTS_OR_INDEX_TEXT"
            elif passage.get("legal_reference"):
                reason = "PROVISION_WITHOUT_SUPPORTED_QUESTION_PATTERN"
            elif not normalize_text(evidence):
                reason = "EMPTY_OR_NON_TEXT_PASSAGE"
            else:
                reason = "NO_SUPPORTED_QUESTION_PATTERN"
            no_candidate_passages.append({
                "passage_id": passage.get("passage_id", ""),
                "source_filename": passage.get("source_filename", ""),
                "page": passage.get("page", ""),
                "legal_reference": passage.get("legal_reference") or "",
                "reason": reason,
                "evidence": evidence,
            })
            continue
        raw_candidates.extend(generated)

    valid_candidates = []
    invalid_candidates = []
    invalid_passages = {}
    valid_passage_ids = set()
    for candidate in raw_candidates:
        validation_flags = validate_candidate(candidate, source_page_texts)
        if validation_flags:
            candidate["validation_flags"] = validation_flags
            candidate["duplicate_flags"] = []
            candidate["selection_flags"] = []
            invalid_candidates.append(candidate)
            invalid_passages.setdefault(str(candidate["passage_id"]), candidate)
            failures.append(
                f"{candidate['candidate_id']}: {'; '.join(validation_flags)}"
            )
        else:
            candidate["validation_flags"] = []
            candidate["duplicate_flags"] = []
            candidate["selection_flags"] = []
            valid_candidates.append(candidate)
            valid_passage_ids.add(str(candidate["passage_id"]))

    already_reported_passages = {
        str(row["passage_id"])
        for row in no_candidate_passages
    }
    for passage_id, candidate in invalid_passages.items():
        if (
            passage_id not in valid_passage_ids
            and passage_id not in already_reported_passages
        ):
            no_candidate_passages.append({
                "passage_id": passage_id,
                "source_filename": candidate["source_filename"],
                "page": candidate["page"],
                "legal_reference": candidate["legal_reference"] or "",
                "reason": "ALL_CANDIDATES_REJECTED_BY_DETERMINISTIC_VALIDATION",
                "evidence": candidate["evidence"],
            })

    deduplicated, duplicate_rejections = deduplicate_candidates(valid_candidates)
    raw_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in raw_candidates
    }
    if len(raw_by_id) != len(raw_candidates):
        raise RuntimeError("Generated candidate IDs are not unique within this run.")
    for row in duplicate_rejections:
        row["flag"] = row["reason"]
        duplicate_candidate = raw_by_id.get(str(row["candidate_id"]))
        if (
            duplicate_candidate is not None
            and row["reason"] not in duplicate_candidate["duplicate_flags"]
        ):
            duplicate_candidate["duplicate_flags"].append(row["reason"])
    selected = select_balanced_candidates(deduplicated, candidate_limit, seed)
    omitted_by_candidate_limit = len(deduplicated) - len(selected)
    selected_ids = {str(row["candidate_id"]) for row in selected}
    for row in selected:
        row["selection_flags"].append("SELECTED_WITHIN_CANDIDATE_LIMIT")
    for row in deduplicated:
        if str(row["candidate_id"]) not in selected_ids:
            if "NOT_SELECTED_BY_CANDIDATE_LIMIT" not in row["selection_flags"]:
                row["selection_flags"].append("NOT_SELECTED_BY_CANDIDATE_LIMIT")
    for row in duplicate_rejections:
        if row["duplicate_of"] not in selected_ids:
            row["flag"] += "; duplicate target omitted by candidate limit"

    return {
        "raw_candidates": raw_candidates,
        "invalid_candidates": invalid_candidates,
        "deduplicated": deduplicated,
        "duplicate_rejections": duplicate_rejections,
        "selected": selected,
        "failures": failures,
        "omitted_by_candidate_limit": omitted_by_candidate_limit,
        "no_candidate_passages": no_candidate_passages,
    }


def _run_corpus(
    output_dir: Path,
    seed: int,
    candidate_limit: int,
    sample_pages_per_source: int | None = None,
) -> dict[str, Any]:
    passages = []
    source_page_texts: dict[tuple[str, int], str] = {}
    page_counts = {}
    usable_counts = {}
    empty_counts = {}

    for filename, pdf_path in SOURCE_PDFS.items():
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Required source PDF is missing: {pdf_path}")
        with pymupdf.open(pdf_path) as document:
            total_pages = len(document)
        page_limit = (
            min(total_pages, sample_pages_per_source)
            if sample_pages_per_source is not None
            else total_pages
        )
        page_texts = extract_selected_pdf_pages(
            pdf_path,
            list(range(1, page_limit + 1)),
        )
        source_page_texts.update({
            (filename, page_number): text
            for page_number, text in page_texts.items()
        })
        usable = []
        empty = 0
        for page_number, page_text in page_texts.items():
            if not page_text.strip():
                empty += 1
                continue
            usable.append(
                split_page_into_passages(filename, page_number, page_text)
            )
        page_counts[filename] = page_limit
        usable_counts[filename] = page_limit - empty
        empty_counts[filename] = empty
        passages.extend(
            passage
            for page_passages in usable
            for passage in page_passages
        )

    generated = generate_from_passages(
        passages,
        source_page_texts,
        candidate_limit,
        seed,
    )
    settings = {
        "mode": (
            "bounded_sample"
            if sample_pages_per_source is not None
            else "full_corpus"
        ),
        "seed": seed,
        "candidate_limit": candidate_limit,
        "selection_policy": (
            "Retain all distinct validated candidates up to the configured "
            "limit. When the pool exceeds the limit, select proportionally by "
            "observed source and question-type availability; no exact split "
            "is imposed."
        ),
        "near_duplicate_question_threshold": NEAR_DUPLICATE_THRESHOLD,
        "pages_processed_by_source": page_counts,
        "usable_pages_by_source": usable_counts,
        "empty_pages_by_source": empty_counts,
        "passages_processed_by_source": dict(Counter(
            str(passage["source_filename"]) for passage in passages
        )),
        "raw_candidate_count": len(generated["raw_candidates"]),
        "invalid_candidate_count": len(generated["invalid_candidates"]),
        "omitted_by_candidate_limit": generated["omitted_by_candidate_limit"],
        "duplicate_candidates_removed": len(generated["duplicate_rejections"]),
        "candidate_limit_reached": (
            candidate_limit > 0
            and len(generated["selected"]) >= candidate_limit
        ),
        "target_500_reached": len(generated["selected"]) >= 500,
        "candidate_shortfall_from_target_500": max(
            0,
            500 - len(generated["selected"]),
        ),
        "candidate_pool_distribution": {
            "raw_by_source": dict(Counter(
                str(row["source_filename"])
                for row in generated["raw_candidates"]
            )),
            "raw_by_question_type": dict(Counter(
                str(row["question_type"])
                for row in generated["raw_candidates"]
            )),
            "deduplicated_by_source": dict(Counter(
                str(row["source_filename"])
                for row in generated["deduplicated"]
            )),
            "deduplicated_by_question_type": dict(Counter(
                str(row["question_type"])
                for row in generated["deduplicated"]
            )),
        },
        "page_provenance": "PyMuPDF page.get_text('text', sort=True), 1-based page number",
        "reference_policy": (
            "Only supported explicit numbered provision headings; uncertain "
            "references remain null and UNCONFIRMED."
        ),
    }
    return write_run_outputs(
        output_dir,
        generated["selected"],
        generated["failures"],
        settings,
        generated["duplicate_rejections"],
        generated["raw_candidates"],
        generated["no_candidate_passages"],
        _compare_stage12(generated),
    )


def _timestamped_output_dir(seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "reports" / f"qa_candidate_generation_{stamp}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run only the two-passage, two-source pilot; never processes the full PDFs.",
    )
    parser.add_argument(
        "--sample-pages-per-source",
        type=int,
        help="Run a bounded first-N-pages-per-PDF sample, for validation before full mode.",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Process every page in both PDFs.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="Maximum retained candidates after validation and deduplication.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional fresh output directory; existing directories are never overwritten.",
    )
    args = parser.parse_args()
    if args.candidate_limit < 0:
        parser.error("--candidate-limit must be zero or greater.")
    modes = sum((
        bool(args.pilot),
        args.sample_pages_per_source is not None,
        bool(args.full_corpus),
    ))
    if modes != 1:
        parser.error("Choose exactly one of --pilot, --sample-pages-per-source, or --full-corpus.")
    output_dir = args.output_dir or _timestamped_output_dir(args.seed)
    if args.pilot:
        manifest = _run_pilot(output_dir)
    else:
        if (
            args.sample_pages_per_source is not None
            and args.sample_pages_per_source < 1
        ):
            parser.error("--sample-pages-per-source must be positive.")
        manifest = _run_corpus(
            output_dir,
            args.seed,
            args.candidate_limit,
            args.sample_pages_per_source,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
