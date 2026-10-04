"""Review Stage 16 legal QA forms one at a time without editing their source."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.review_first_batch_legal_qa import (  # noqa: E402
    DECISIONS,
    _candidate_form_mismatches,
    _display_path,
    validate_decision_form,
)

DEFAULT_WORKSPACE = ROOT / "reports" / "legal_qa_individual_review_20261004T094008Z"
DEFAULT_BATCH = (
    ROOT
    / "reports"
    / "legal_qa_review_prioritization_20261004T093327Z"
    / "first_review_batch.csv"
)
DEFAULT_STAGE15 = (
    ROOT
    / "reports"
    / "legal_qa_review_prioritization_20261004T093327Z"
    / "prioritized_candidates.csv"
)
DECISIONS_OUTPUT = "decision_forms"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_form(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Review form must be a JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _timestamped_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "reports" / f"legal_qa_review_session_{stamp}"


def _format_or_missing(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "[not confirmed / not supplied]"


def render_candidate(
    form: Mapping[str, Any],
    position: int,
    total: int,
    pdf_path: Path,
) -> str:
    candidate = form["candidate"]
    duplicate_flags = _format_or_missing(candidate.get("duplicate_flags"))
    template_flag = (
        "YES — possible repeated answer/template"
        if str(candidate.get("answer_template_repetition", "")).casefold() == "true"
        else "No template repetition flag"
    )
    return "\n".join([
        "",
        "=" * 78,
        f"Candidate {position} of {total}",
        f"ID: {candidate.get('candidate_id', '')}",
        "=" * 78,
        "QUESTION:",
        str(candidate.get("question", "")),
        "",
        "PROPOSED ANSWER:",
        str(candidate.get("proposed_answer", "")),
        "",
        "SUPPORTING PASSAGE (full):",
        str(candidate.get("supporting_passage", "")),
        "",
        f"SOURCE PDF: {pdf_path}",
        f"PDF page: {candidate.get('pdf_page', '')}",
        f"Article/Section reference: "
        f"{_format_or_missing(candidate.get('legal_reference'))}",
        f"Reference status/flags: "
        f"{_format_or_missing(candidate.get('reference_status'))}; "
        f"{_format_or_missing(candidate.get('reference_confidence_flags'))}",
        f"Evidence traceability flag: "
        f"{_format_or_missing(candidate.get('evidence_status'))}",
        f"Question type: {_format_or_missing(candidate.get('question_type'))}",
        f"Duplicate flags: {duplicate_flags}",
        f"Related candidate IDs: "
        f"{_format_or_missing(candidate.get('duplicate_candidate_ids'))}",
        f"Answer-template warning: {template_flag}",
        "",
        "CHECK THIS CANDIDATE IN THE ORIGINAL PDF:",
        f"1. Open the PDF above in a PDF reader; go to PDF page "
        f"{candidate.get('pdf_page', '')} (not a printed page label).",
        "2. Use the PDF reader's Find (Ctrl+f) for the cited Article/Section "
        "heading. If the reference is blank "
        "or unconfirmed, do not guess it; search a distinctive phrase from the "
        "supporting passage and identify the provision from the original text.",
        "3. Compare the complete supporting passage with the provision and "
        "its surrounding context.",
        "4. Check that the proposed answer is actually supported and does not "
        "omit a condition, exception, or limitation.",
        "5. Consider duplicate/template flags; they are warnings, not grounds "
        "for automatic rejection.",
        "",
        "APPROVE only if you personally verified answer support in the passage "
        "and checked the legal reference against the original legal source. "
        "Keep uncertain interpretations or unconfirmed references PENDING.",
        "REJECT only when you can state a reason. EDIT_AND_RECHECK requires a "
        "proposed answer revision and does not approve the record.",
        "Automated matching and scores are not proof of legal correctness.",
        "=" * 78,
    ])


def _read_multiline(
    prompt: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    output_fn(prompt)
    output_fn("Enter text; finish by entering a single period (.) on its own line.")
    lines = []
    while True:
        line = input_fn("")
        if line == ".":
            return "\n".join(lines).strip()
        lines.append(line)


def collect_review_decision(
    form: Mapping[str, Any],
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Prompt until the reviewer supplies a valid decision; never infer one."""
    result = copy.deepcopy(dict(form))
    candidate = result["candidate"]
    result.setdefault("review", {})
    review = result["review"]
    original_answer = str(candidate.get("proposed_answer") or "")

    while True:
        output_fn("Choose decision: APPROVE / REJECT / EDIT_AND_RECHECK / PENDING")
        decision = input_fn("Decision: ").strip().upper()
        if decision not in DECISIONS:
            output_fn("Invalid choice. Choose one of the four listed values.")
            continue
        review.update({
            "decision": decision,
            "reviewer": "",
            "reason": "",
            "answer_supported_by_passage_confirmation": "NO",
            "legal_reference_checked_against_original_source": "NO",
            "edited_answer_rechecked_confirmation": "NO",
            "edited_answer": "",
            "reviewed_legal_reference": "",
            "notes": "",
        })
        if decision == "PENDING":
            review_errors = validate_decision_form(
                result,
                {"candidate_id": candidate.get("candidate_id")},
            )
            if review_errors:
                output_fn("This pending decision did not pass form validation.")
                continue
            return result

        review["reviewer"] = input_fn("Reviewer name or identifier: ").strip()
        review["reason"] = input_fn("Reason for this decision: ").strip()
        if decision == "EDIT_AND_RECHECK":
            review["edited_answer"] = _read_multiline(
                "Enter the proposed revised answer:",
                input_fn,
                output_fn,
            )
        elif decision == "APPROVE":
            if (
                not str(candidate.get("legal_reference") or "").strip()
                or candidate.get("reference_status") == "UNCONFIRMED"
            ):
                review["reviewed_legal_reference"] = input_fn(
                    "If you checked the original source and identified an exact "
                    "Article/Section, enter it as printed; otherwise leave blank: "
                ).strip()
            review["answer_supported_by_passage_confirmation"] = (
                input_fn(
                    "Have you checked that the answer is supported by the "
                    "passage? Enter YES or NO: "
                ).strip().upper()
            )
            review["legal_reference_checked_against_original_source"] = (
                input_fn(
                    "Have you checked the legal reference against the original "
                    "legal source? Enter YES or NO: "
                ).strip().upper()
            )
            if (
                review["answer_supported_by_passage_confirmation"] == "YES"
                and review["legal_reference_checked_against_original_source"] == "YES"
                and review["edited_answer"]
            ):
                review["edited_answer_rechecked_confirmation"] = (
                    input_fn(
                        "If this answer was edited, did you recheck the edit "
                        "against the source? Enter YES or NO: "
                    ).strip().upper()
                )

        errors = validate_decision_form(
            result,
            {"candidate_id": candidate.get("candidate_id")},
        )
        if errors:
            output_fn("This choice cannot be saved as entered:")
            for error in errors:
                output_fn(f"- {error}")
            output_fn(
                "Choose PENDING or enter a completed decision with all "
                "required information."
            )
            continue
        if decision == "APPROVE" and review.get("edited_answer"):
            if review["edited_answer"].strip() == original_answer.strip():
                # The explicit recheck confirmation is unnecessary when no edit occurred.
                review["edited_answer_rechecked_confirmation"] = "NO"
        return result


def _load_candidate_order(
    workspace: Path,
    batch_csv: Path,
    prioritized_csv: Path,
) -> list[tuple[dict[str, Any], Path, dict[str, str]]]:
    batch = read_csv(batch_csv)
    priority_rows = {
        row["candidate_id"]: row for row in read_csv(prioritized_csv)
    }
    candidates = []
    for row in sorted(batch, key=lambda item: int(item.get("batch_order") or 0)):
        candidate_id = row["candidate_id"]
        path = workspace / "candidate_forms" / f"{candidate_id}.json"
        form = read_form(path)
        if candidate_id not in priority_rows:
            raise ValueError(f"Candidate is missing from priority report: {candidate_id}")
        mismatches = _candidate_form_mismatches(
            form.get("candidate", {}),
            priority_rows[candidate_id],
        )
        if mismatches:
            raise ValueError(
                f"Original form {candidate_id} differs from Stage 15 source: "
                + ", ".join(mismatches)
            )
        if form.get("review", {}).get("decision") != "PENDING":
            raise ValueError(f"Original review form is not blank/pending: {candidate_id}")
        candidates.append((form, path, row))
    if len(candidates) != 50:
        raise ValueError(f"Expected 50 review forms; found {len(candidates)}.")
    return candidates


def _write_session_instructions(path: Path, output_dir: Path, batch_csv: Path) -> None:
    text = (
        "# Interactive legal QA review session\n\n"
        "Run the session from the repository root in Git Bash. To review the "
        "first candidate, use:\n\n"
        "```bash\n"
        "cd /d/Legal_AI_Projects/legal_ai_rag\n"
        "./venv/Scripts/python.exe scripts/interactive_legal_qa_review.py "
        "--start-at 1\n"
        "```\n\n"
        "Each candidate is displayed individually. The helper writes submitted "
        "decisions under `decision_forms/` in this new session directory; it "
        "does not edit the original Stage 16 forms. The session output path is "
        f"`{_display_path(output_dir)}`. It also contains a summary after a "
        "session completes.\n\n"
        "To import and validate decisions later, use the existing importer "
        "with a new, unused output directory:\n\n"
        "```bash\n"
        "cd /d/Legal_AI_Projects/legal_ai_rag\n"
        "./venv/Scripts/python.exe scripts/review_first_batch_legal_qa.py "
        "import --forms-dir "
        f"{_display_path(output_dir / DECISIONS_OUTPUT)} "
        f"--batch-csv {_display_path(batch_csv)} "
        "--output-dir reports/legal_qa_review_import_<new-unique-name>\n"
        "```\n\n"
        "The importer remains authoritative: it validates explicit decisions "
        "and source/evidence constraints. Automated checks do not establish "
        "legal correctness, and no dataset split is created by this workflow.\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def run_review_session(
    workspace: Path,
    batch_csv: Path,
    prioritized_csv: Path,
    output_dir: Path,
    start_at: int = 1,
    limit: int | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    if start_at < 1:
        raise ValueError("start_at must be at least 1.")
    candidates = _load_candidate_order(workspace, batch_csv, prioritized_csv)
    stop = len(candidates) if limit is None else min(
        len(candidates),
        start_at - 1 + max(0, limit),
    )
    if start_at > len(candidates) + 1:
        raise ValueError("start_at is past the end of the 50-candidate batch.")

    decision_dir = output_dir / DECISIONS_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=False)
    decision_dir.mkdir()
    _write_session_instructions(
        output_dir / "README.md",
        output_dir,
        batch_csv,
    )
    for index in range(start_at - 1, stop):
        form, source_path, batch_row = candidates[index]
        candidate = form["candidate"]
        source_pdf = ROOT / "data" / str(candidate.get("source_document") or "")
        output_fn(render_candidate(form, index + 1, len(candidates), source_pdf))
        reviewed = collect_review_decision(form, input_fn, output_fn)
        _write_json_exclusive(
            decision_dir / source_path.name,
            reviewed,
        )

    saved_forms = [
        read_form(path)
        for path in decision_dir.glob("*.json")
    ]
    counts = Counter(
        str(form.get("review", {}).get("decision") or "PENDING")
        for form in saved_forms
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_workspace": _display_path(workspace),
        "start_at": start_at,
        "requested_limit": limit,
        "saved_form_count": len(saved_forms),
        "decision_counts": {
            decision: counts[decision]
            for decision in ("APPROVE", "REJECT", "EDIT_AND_RECHECK", "PENDING")
        },
        "decisions_directory": DECISIONS_OUTPUT,
        "import_command_example": (
            ".\\\\venv\\\\Scripts\\\\python.exe scripts\\\\review_first_batch_legal_qa.py "
            "import --forms-dir <session-output>\\\\decision_forms "
            f"--batch-csv {_display_path(batch_csv)} "
            "--output-dir <new-empty-import-output>"
        ),
        "legal_correctness_claim": (
            "None. Reviewer decisions and importer checks do not independently "
            "establish legal correctness."
        ),
    }
    _write_json_exclusive(output_dir / "session_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Untouched Stage 16 individual-form workspace.",
    )
    parser.add_argument("--batch-csv", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--prioritized-csv", type=Path, default=DEFAULT_STAGE15)
    parser.add_argument("--output-dir", type=Path, default=_timestamped_output_dir())
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit cannot be negative.")
    summary = run_review_session(
        args.workspace,
        args.batch_csv,
        args.prioritized_csv,
        args.output_dir,
        args.start_at,
        args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
