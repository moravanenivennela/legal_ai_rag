"""Conservative second-pass evidence triage for supported legal QA records."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.legal_qa_audit_core import (
    ROOT,
    SOURCE_PDFS,
    _find_excerpt,
    _heading_index,
    _reference_check,
    load_pdf_corpus,
    normalize_text,
    source_key,
)


DEFAULT_AUDIT_DIR = ROOT / "reports" / "legal_qa_full_audit_20261004T143722Z"
OUTPUT_PREFIX = "legal_qa_source_evidence_review_"
QUALIFICATION_RE = re.compile(
    r"\b(?:provided that|unless|except|subject to|notwithstanding|"
    r"shall not|may not|only if|in the case of|whereas)\b",
    re.IGNORECASE,
)
OCR_ARTIFACT_RE = re.compile(r"\ufffd|\w\?\w|\?{2,}")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _json_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return value["records"]
    raise ValueError(f"Expected a JSON object containing records: {path}")


def _validate_audit_bundle(audit_dir: Path) -> list[dict[str, Any]]:
    required = (
        "full_audit.json",
        "full_audit.csv",
        "supported_candidates.json",
        "audit_metrics.json",
        "run_manifest.json",
    )
    missing = [name for name in required if not (audit_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Audit bundle is missing required files: {', '.join(missing)}"
        )

    manifest = json.loads(
        (audit_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (audit_dir / "audit_metrics.json").read_text(encoding="utf-8")
    )
    full_audit = json.loads(
        (audit_dir / "full_audit.json").read_text(encoding="utf-8")
    )
    supported_bundle = json.loads(
        (audit_dir / "supported_candidates.json").read_text(encoding="utf-8")
    )
    if manifest.get("completed") is not True:
        raise ValueError("The source audit manifest is not marked completed.")
    if manifest.get("input_fingerprint") != metrics.get("input_fingerprint"):
        raise ValueError("Audit manifest and metrics fingerprints do not match.")
    records = full_audit.get("records")
    if not isinstance(records, list):
        raise ValueError("full_audit.json does not contain a records list.")
    if len(records) != full_audit.get("count_unique_candidates"):
        raise ValueError("Full audit record count does not match its envelope.")

    csv_rows: list[dict[str, str]] = []
    with (audit_dir / "full_audit.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        csv_rows.extend(csv.DictReader(stream))
    full_ids = {str(row.get("candidate_id") or "") for row in records}
    csv_ids = {str(row.get("candidate_id") or "") for row in csv_rows}
    if len(full_ids) != len(records) or full_ids != csv_ids:
        raise ValueError("Full JSON and CSV candidate IDs are not identical/unique.")

    supported = [
        row for row in records
        if row.get("automated_status") == "SUPPORTED_CANDIDATE"
    ]
    supported_bundle_records = supported_bundle.get("records")
    if not isinstance(supported_bundle_records, list):
        raise ValueError("supported_candidates.json does not contain records.")
    supported_ids = {str(row.get("candidate_id") or "") for row in supported}
    bundled_ids = {
        str(row.get("candidate_id") or "") for row in supported_bundle_records
    }
    if (
        supported_ids != bundled_ids
        or len(supported) != supported_bundle.get("count")
        or len(supported) != metrics.get("supported_candidates")
    ):
        raise ValueError("Supported-candidate input counts or IDs do not match.")
    return supported


def review_candidate_evidence(
    record: Mapping[str, Any],
    pages_by_source: Mapping[str, Sequence[str] | None],
    pdf_errors: Mapping[str, str] | None = None,
    headings_by_source: Mapping[str, Mapping[str, set[int]]] | None = None,
) -> dict[str, Any]:
    """Recheck literal evidence in local PDF text without asserting legal meaning."""
    pdf_errors = pdf_errors or {}
    headings_by_source = headings_by_source or _heading_index(pages_by_source)
    source = source_key(str(record.get("source_document") or ""))
    page_value = record.get("declared_pdf_page")
    try:
        declared_page = int(page_value)
    except (TypeError, ValueError):
        declared_page = None

    page_text = ""
    adjacent_context: dict[str, str] = {}
    if (
        source
        and pages_by_source.get(source) is not None
        and declared_page is not None
    ):
        pages = pages_by_source[source]
        assert pages is not None
        if 1 <= declared_page <= len(pages):
            page_text = pages[declared_page - 1]
            for page_number in range(
                max(1, declared_page - 1),
                min(len(pages), declared_page + 1) + 1,
            ):
                adjacent_context[str(page_number)] = pages[page_number - 1]

    supplied_excerpt = str(record.get("exact_source_evidence_excerpt") or "")
    answer = str(record.get("original_answer") or "")
    question = str(record.get("original_question") or "")
    reference = str(record.get("original_reference") or "")
    pdf_excerpt = _find_excerpt(page_text, supplied_excerpt) if page_text else ""
    answer_excerpt = _find_excerpt(pdf_excerpt, answer) if pdf_excerpt else ""
    excerpt_reverified = bool(supplied_excerpt and pdf_excerpt)
    answer_verbatim_in_excerpt = bool(answer and answer_excerpt)
    reference_status, reference_reason, reference_pages = _reference_check(
        reference,
        source,
        declared_page,
        pages_by_source.get(source) if source else None,
        headings_by_source.get(source, {}) if source else {},
        supplied_excerpt,
    )

    flags: list[str] = []
    if not source or pages_by_source.get(source) is None:
        flags.append("SOURCE_PDF_UNAVAILABLE")
    if declared_page is None or not page_text:
        flags.append("DECLARED_PDF_PAGE_UNAVAILABLE")
    if not excerpt_reverified:
        flags.append("SOURCE_EXCERPT_NOT_REVERIFIED_ON_DECLARED_PAGE")
    if not answer_verbatim_in_excerpt:
        flags.append("ANSWER_NOT_FOUND_VERBATIM_WITHIN_SOURCE_EXCERPT")
    if reference_status != "REFERENCE_MATCHED_PDF_HEADING":
        flags.append("REFERENCE_NOT_CONFIRMED_AGAINST_PDF_HEADING")

    inspected_text = "\n".join(
        (question, answer, reference, supplied_excerpt, page_text)
    )
    if OCR_ARTIFACT_RE.search(inspected_text):
        flags.append("POSSIBLE_OCR_OR_REPLACEMENT_CHARACTER_CORRUPTION")
    if CONTROL_CHARACTER_RE.search(inspected_text):
        flags.append("CONTROL_CHARACTER_EXTRACTION_ARTIFACT")

    normalized_excerpt = normalize_text(supplied_excerpt)
    normalized_answer = normalize_text(answer)
    context_qualifiers = {
        match.group(0).casefold()
        for match in QUALIFICATION_RE.finditer(page_text)
    }
    excerpt_qualifiers = {
        match.group(0).casefold()
        for match in QUALIFICATION_RE.finditer(supplied_excerpt)
    }
    if context_qualifiers - excerpt_qualifiers:
        flags.append("POTENTIAL_QUALIFICATION_OUTSIDE_SUPPLIED_EXCERPT")

    existing_issues = list(record.get("issues") or [])
    if existing_issues:
        flags.append("PRIOR_AUDIT_ISSUES_PRESENT")
    flags.append("PROVISION_CONTEXT_COMPLETENESS_NOT_ESTABLISHED")
    flags.append("QUESTION_ANSWER_LEGAL_ALIGNMENT_REQUIRES_HUMAN_REVIEW")
    if not normalized_excerpt or not normalized_answer:
        flags.append("SOURCE_OR_ANSWER_TEXT_EMPTY")

    exact_source_text_match = bool(
        excerpt_reverified and answer_verbatim_in_excerpt
    )
    hard_or_ambiguous_flags = {
        "SOURCE_PDF_UNAVAILABLE",
        "DECLARED_PDF_PAGE_UNAVAILABLE",
        "SOURCE_EXCERPT_NOT_REVERIFIED_ON_DECLARED_PAGE",
        "ANSWER_NOT_FOUND_VERBATIM_WITHIN_SOURCE_EXCERPT",
        "REFERENCE_NOT_CONFIRMED_AGAINST_PDF_HEADING",
        "POSSIBLE_OCR_OR_REPLACEMENT_CHARACTER_CORRUPTION",
        "CONTROL_CHARACTER_EXTRACTION_ARTIFACT",
        "POTENTIAL_QUALIFICATION_OUTSIDE_SUPPLIED_EXCERPT",
        "PRIOR_AUDIT_ISSUES_PRESENT",
        "SOURCE_OR_ANSWER_TEXT_EMPTY",
    }
    triage_status = (
        "SOURCE_MATCH_REQUIRES_REVIEW"
        if hard_or_ambiguous_flags.intersection(flags)
        else "EXACT_SOURCE_EVIDENCE_REQUIRES_REVIEW"
    )
    reasoning = (
        "The answer's complete normalized token sequence was found within the "
        "supplied excerpt, and that excerpt was re-found on the declared PDF "
        "page. This establishes literal source traceability only; it does not "
        "establish that the answer correctly responds to the question, captures "
        "all legal qualifications, or states current law."
        if exact_source_text_match
        else
        "The answer and supplied excerpt did not satisfy all literal-source "
        "checks against the declared PDF page. The listed flags identify the "
        "remaining evidence gaps; do not infer support from word overlap."
    )
    suggested_question = str(record.get("suggested_question") or "")
    suggested_answer = str(record.get("suggested_answer") or "")
    suggested_reference = str(record.get("suggested_reference") or "")
    return {
        "candidate_id": str(record.get("candidate_id") or ""),
        "source_audit_status": str(record.get("automated_status") or ""),
        "source_match_triage_status": triage_status,
        "exact_source_text_match": exact_source_text_match,
        "answer_verbatim_in_source_excerpt": answer_verbatim_in_excerpt,
        "answer_to_question_semantic_support": "NOT_ESTABLISHED_AUTOMATICALLY",
        "review_disposition": "REQUIRES_INDEPENDENT_HUMAN_LEGAL_REVIEW",
        "human_approval_status": "NOT_APPROVED_BY_AUTOMATION",
        "legal_verification_status": "NOT_LEGALLY_VERIFIED",
        "current_law_status": "NOT_CHECKED",
        "potential_outdated_provision_review_required": True,
        "potential_outdated_provision_note": (
            "The supplied PDF was checked as provided; amendments, "
            "commencement, and current-law status were not independently checked."
        ),
        "original_question": question,
        "original_answer": answer,
        "original_reference": reference,
        "suggested_question": suggested_question,
        "suggested_answer": suggested_answer,
        "suggested_reference": suggested_reference,
        "source_document": source or str(record.get("source_document") or ""),
        "declared_pdf_page": declared_page,
        "reverified_pdf_page": declared_page if page_text else None,
        "source_excerpt_from_audit": supplied_excerpt,
        "exact_matching_source_excerpt": pdf_excerpt,
        "answer_excerpt_from_source": answer_excerpt,
        "surrounding_pdf_context_by_page": adjacent_context,
        "reference_match_result": reference_status,
        "reference_match_reason": reference_reason,
        "reference_heading_pages": reference_pages,
        "audit_reasoning": str(record.get("reason") or ""),
        "second_pass_reasoning": reasoning,
        "existing_audit_issues": existing_issues,
        "review_flags": sorted(set(flags)),
        "provision_context_completeness": "NOT_ESTABLISHED_AUTOMATICALLY",
        "source_pdf_error": pdf_errors.get(source, "") if source else "",
        "source_audit_confidence": record.get("confidence"),
        "source_audit_confidence_scope": record.get("confidence_scope", ""),
        "source_audit_duplicate_group_id": record.get("duplicate_group_id", ""),
    }


def write_review_report(
    records: Sequence[Mapping[str, Any]],
    audit_dir: Path,
    output_root: Path,
    data_dir: Path,
) -> Path:
    pages, pdf_errors, pdf_paths = load_pdf_corpus(data_dir)
    headings = _heading_index(pages)
    reviewed = [
        review_candidate_evidence(row, pages, pdf_errors, headings)
        for row in records
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"{OUTPUT_PREFIX}{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_audit_directory": str(audit_dir),
        "source_pdf_paths": pdf_paths,
        "record_count": len(reviewed),
        "notice": (
            "Automated evidence triage only. No record is human-approved, "
            "legally verified, or eligible for fine-tuning based on this report."
        ),
        "records": reviewed,
    }
    (output_dir / "source_evidence_review.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "candidate_id",
        "source_audit_status",
        "source_match_triage_status",
        "exact_source_text_match",
        "answer_verbatim_in_source_excerpt",
        "answer_to_question_semantic_support",
        "review_disposition",
        "human_approval_status",
        "legal_verification_status",
        "current_law_status",
        "potential_outdated_provision_review_required",
        "original_question",
        "original_answer",
        "original_reference",
        "suggested_question",
        "suggested_answer",
        "suggested_reference",
        "source_document",
        "declared_pdf_page",
        "reverified_pdf_page",
        "source_excerpt_from_audit",
        "exact_matching_source_excerpt",
        "answer_excerpt_from_source",
        "surrounding_pdf_context_by_page",
        "reference_match_result",
        "reference_match_reason",
        "reference_heading_pages",
        "audit_reasoning",
        "second_pass_reasoning",
        "existing_audit_issues",
        "review_flags",
        "provision_context_completeness",
        "potential_outdated_provision_note",
        "source_pdf_error",
        "source_audit_confidence",
        "source_audit_confidence_scope",
        "source_audit_duplicate_group_id",
    ]
    with (output_dir / "source_evidence_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in reviewed:
            writer.writerow({
                key: (
                    json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                )
                for key in csv_fields
            })

    exact_count = sum(bool(row["exact_source_text_match"]) for row in reviewed)
    triage_counts: dict[str, int] = {}
    for row in reviewed:
        key = str(row["source_match_triage_status"])
        triage_counts[key] = triage_counts.get(key, 0) + 1
    ambiguity_count = triage_counts.get("SOURCE_MATCH_REQUIRES_REVIEW", 0)
    summary = [
        "# Supported-candidate source-evidence triage",
        "",
        f"- Candidates processed: {len(reviewed)}",
        f"- Candidates with exact source-text evidence: {exact_count}",
        f"- Candidates classified `SOURCE_MATCH_REQUIRES_REVIEW`: {ambiguity_count}",
        f"- Source audit directory: `{audit_dir}`",
        "",
        "## Scope and interpretation",
        "",
        "This report rechecks source excerpts against the supplied local PDFs. "
        "An exact source-text match means the complete normalized answer token "
        "sequence was found contiguously inside the supplied excerpt, and that "
        "excerpt was re-found on the declared PDF page. Punctuation and "
        "whitespace differences are ignored for this token-sequence comparison.",
        "",
        "Even an exact textual match does not establish that the answer correctly "
        "answers the question, preserves every qualification, is legally correct, "
        "or reflects current law. **All records remain pending independent human "
        "legal review; none is legally verified, approved, or ready for "
        "fine-tuning.**",
        "",
        "Records with missing or ambiguous evidence, reference mismatches, "
        "possible OCR corruption, possible qualifications outside the excerpt, "
        "or prior audit issues use `SOURCE_MATCH_REQUIRES_REVIEW`. Context "
        "completeness, amendments, and commencement/current-law status are not "
        "established automatically.",
        "",
        "## Triage status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    summary.extend(
        f"| `{status}` | {count} |"
        for status, count in sorted(triage_counts.items())
    )
    summary.extend([
        "",
        "## Important limitation",
        "",
        "This is automated evidence triage, not legal verification. Suggested "
        "wording is kept separate from original text and is not approval.",
        "",
    ])
    (output_dir / "summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recheck source evidence for SUPPORTED_CANDIDATE records without "
            "approving or legally verifying them."
        )
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    supported = _validate_audit_bundle(args.audit_dir)
    output_dir = write_review_report(
        supported,
        args.audit_dir,
        args.output_root,
        args.data_dir,
    )
    print(f"Candidates processed: {len(supported)}")
    print(f"Source-evidence review written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
