"""Browser dashboard for batch review of the 21 Stage 23 legal QA candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from scripts.stage25_review_dashboard_data import (
    DECISIONS,
    EXPORT_NOTICE,
    ROOT,
    build_csv_export,
    build_external_review_report,
    build_json_export,
    build_saved_decisions,
    load_dashboard_records,
    write_exports,
    write_saved_decisions,
)

_COPY_REPORT_COMPONENT = st.components.v2.component(
    "stage25_copy_external_review_report",
    html=(
        '<button id="copy-report" type="button">Copy all 21 candidate report</button> '
        '<span id="copy-status" role="status" aria-live="polite"></span>'
    ),
    js="""
export default function ({ data, parentElement }) {
  const button = parentElement.querySelector("#copy-report")
  const status = parentElement.querySelector("#copy-status")
  if (!button || !status) return

  button.onclick = async () => {
    try {
      await navigator.clipboard.writeText(data.report)
      status.textContent = "Copied to clipboard."
    } catch {
      status.textContent =
        "Clipboard access was blocked. Expand the report preview and copy it there."
    }
  }
}
""",
)


def _show_text(label: str, value: str) -> None:
    st.markdown(f"**{label}**")
    st.text(value if value else "[not supplied]")


def _set_download_message(export_kind: str) -> None:
    st.session_state["stage25_action_message"] = (
        "success",
        f"{export_kind} download started. Check your browser's download location.",
    )


def _render_candidate(record: dict, position: int) -> None:
    candidate_id = record["candidate_id"]
    with st.container(border=True):
        st.subheader(f"{position}. Candidate {candidate_id}")
        st.caption(
            f"Source: {record['source_document']} | PDF page "
            f"{record['pdf_page_1_based']} | Type: "
            f"{record['question_type'] or 'not supplied'}"
        )

        _show_text("Original question", record["original_question"])
        _show_text("Original answer", record["original_answer"])
        _show_text("Suggested question (AI-prepared, not approved)", record["proposed_question"])
        _show_text("Suggested answer (AI-prepared, not approved)", record["proposed_answer"])
        _show_text("Original legal reference", record["original_reference"])
        _show_text("Proposed legal reference", record["proposed_reference"])
        _show_text(
            "Reference confidence flags",
            record["reference_confidence_flags"],
        )
        _show_text("Evidence traceability status", record["evidence_status"])

        st.markdown("**Stage 16 supporting passage**")
        st.text(record["supporting_passage"])
        st.markdown("**Stage 23 source excerpt**")
        st.text(record["source_provision_excerpt"])
        if record["adjacent_page_excerpt_if_needed"]:
            st.markdown("**Stage 23 adjacent-page excerpt**")
            st.text(record["adjacent_page_excerpt_if_needed"])
        st.markdown("**Extracted original PDF context (full adjacent pages)**")
        for page in record["source_page_context"]:
            st.caption(f"PDF page {page['pdf_page_1_based']}")
            st.text(page["text"])

        _show_text("Stage 23 audit finding", record["source_text_finding"])
        _show_text("Reference/page context", record["reference_page_consistency"])
        _show_text("Audit concerns", record["quality_or_scope_flags"])
        _show_text("Stage 22 evidence notes", record["stage22_evidence_notes"])
        _show_text("Stage 22 correction notes", record["stage22_correction_notes"])
        _show_text("Stage 22 unresolved issues", record["stage22_unresolved_issues"])
        _show_text("Duplicate flags", record["duplicate_flags"])
        st.warning(
            "AI recommendation only; not a legal conclusion: "
            f"{record['ai_recommendation']}"
        )
        st.selectbox(
            "Draft user decision (not saved until you press Save all decisions)",
            options=DECISIONS,
            index=0,
            key=f"stage25_draft_{candidate_id}",
        )


def main() -> None:
    st.set_page_config(
        page_title="Stage 25 Legal QA Review",
        page_icon=":material/rate_review:",
        layout="wide",
    )
    st.title("Stage 25: Review 21 Legal QA Candidates")
    st.warning(
        "Source-text matching and AI recommendations do not establish legal "
        "correctness or current-law status. An APPROVE selection is only a "
        "reviewer selection here; this dashboard does not record the required "
        "reviewer reason and source confirmations, so it is not final-dataset "
        "eligible."
    )

    try:
        if "stage25_dashboard_records" not in st.session_state:
            st.session_state["stage25_dashboard_records"] = load_dashboard_records()
        records = st.session_state["stage25_dashboard_records"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        st.error(f"Could not load the review batch safely: {error}")
        st.stop()

    st.write(
        f"Loaded {len(records)} unique candidates. Every candidate is shown "
        "below in one scrollable page; original text and prior review files "
        "are read-only."
    )
    st.caption(
        "The Stage 19 label is an AI screening suggestion. Dropdown choices "
        "remain drafts until explicitly saved."
    )

    saved_decisions = st.session_state.get("stage25_saved_decisions", {})
    for record in records:
        candidate_id = record["candidate_id"]
        widget_key = f"stage25_draft_{candidate_id}"
        st.session_state.setdefault(
            widget_key,
            saved_decisions.get(candidate_id, "PENDING"),
        )

    for index, record in enumerate(records, start=1):
        _render_candidate(record, index)

    draft_decisions = {
        record["candidate_id"]: st.session_state[
            f"stage25_draft_{record['candidate_id']}"
        ]
        for record in records
    }
    try:
        export_data = build_json_export(records, draft_decisions, saved_decisions)
        json_text = json.dumps(export_data, ensure_ascii=False, indent=2)
        csv_text = build_csv_export(export_data)
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"Could not prepare the complete export: {error}")
        st.stop()

    st.header("Save and export")
    st.info(
        "Downloads contain the full record details, including source passages "
        "and extracted adjacent-page context. Draft choices are marked as "
        "unsaved in exports."
    )
    save_col, export_col = st.columns(2)
    with save_col:
        save_clicked = st.button("Save All Decisions", type="primary")
    with export_col:
        export_clicked = st.button("Export JSON and CSV")

    if save_clicked:
        try:
            payload = build_saved_decisions(records, draft_decisions)
            full_export = build_json_export(
                records,
                draft_decisions,
                saved_decisions=draft_decisions,
            )
            full_export["export_type"] = "SAVED_COMPLETE_CANDIDATE_REVIEW_EXPORT"
            full_export["decision_notice"] = (
                "Dropdown selections were explicitly saved by the user, but do "
                "not constitute completed legal review or final-dataset approval."
            )
            saved_path = write_saved_decisions(
                ROOT / "reports",
                payload,
                full_export,
            )
            st.session_state["stage25_saved_decisions"] = draft_decisions.copy()
            saved_paths = st.session_state.setdefault("stage25_saved_paths", [])
            saved_paths.append(str(saved_path))
            st.session_state["stage25_action_message"] = (
                "success",
                "Saved all 21 decision selections to "
                f"{saved_path} and the full review export to "
                f"{saved_path.parent / 'complete_review_export.json'}.",
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            st.session_state["stage25_action_message"] = (
                "error",
                f"Could not save the 21 decisions: {error}",
            )

    if export_clicked:
        try:
            json_path, csv_path = write_exports(
                ROOT / "reports",
                json_text,
                csv_text,
            )
            st.session_state["stage25_action_message"] = (
                "success",
                f"Exported the full JSON to {json_path} and CSV to {csv_path}.",
            )
        except (OSError, TypeError, ValueError) as error:
            st.session_state["stage25_action_message"] = (
                "error",
                f"Could not export the review records: {error}",
            )

    action_message = st.session_state.get("stage25_action_message")
    if action_message:
        level, message = action_message
        if level == "success":
            st.success(message)
        else:
            st.error(message)

    if st.session_state.get("stage25_saved_paths"):
        st.markdown("**Saved decision snapshots**")
        for path in st.session_state["stage25_saved_paths"]:
            st.code(path, language=None)

    export_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    col_json, col_csv = st.columns(2)
    with col_json:
        st.download_button(
            "Export full review details as JSON",
            data=json_text.encode("utf-8"),
            file_name=f"stage25_legal_qa_review_{export_name}.json",
            mime="application/json",
            key="stage25_download_json",
            on_click=_set_download_message,
            args=("JSON",),
        )
    with col_csv:
        st.download_button(
            "Export full-detail CSV summary",
            data=csv_text.encode("utf-8-sig"),
            file_name=f"stage25_legal_qa_review_{export_name}.csv",
            mime="text/csv",
            key="stage25_download_csv",
            on_click=_set_download_message,
            args=("CSV",),
        )

    st.subheader("Copy the complete 21-candidate report")
    st.write(
        "Copy the full, readable report to paste into your external AI tool. "
        "Candidate IDs and full evidence are included; user selections are "
        "shown separately from saved selections."
    )
    if st.button("Prepare complete copy report"):
        try:
            st.session_state["stage25_external_review_report"] = (
                build_external_review_report(records, draft_decisions)
            )
            st.session_state["stage25_copy_message"] = (
                "success",
                "Prepared the complete report for copying.",
            )
        except (KeyError, TypeError, ValueError) as error:
            st.session_state["stage25_copy_message"] = (
                "error",
                f"Could not prepare the copy report: {error}",
            )

    copy_message = st.session_state.get("stage25_copy_message")
    if copy_message:
        level, message = copy_message
        if level == "success":
            st.success(message)
        else:
            st.error(message)
    external_report = st.session_state.get("stage25_external_review_report")
    if external_report:
        _COPY_REPORT_COMPONENT(
            key="stage25_copy_report",
            data={"report": external_report},
            width="content",
            height=48,
        )
        with st.expander("Preview complete copy report"):
            st.code(external_report, language=None)
    st.caption(EXPORT_NOTICE)


if __name__ == "__main__":
    main()
