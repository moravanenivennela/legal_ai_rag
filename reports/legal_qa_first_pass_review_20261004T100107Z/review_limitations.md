# First-pass review limitations

This report is an AI-assisted, source-text screening of the complete 50-candidate first batch. It is not legal advice, a human review decision, or a legal verification. The original candidate forms, source PDFs, and earlier reports were read only and were not modified.

## What was checked

- Read each candidate question, proposed answer, evidence passage, source filename, declared PDF page, and supplied Article/Section reference.
- Compared the candidate passages with text extracted from the declared pages of `data/constitution_of_india.pdf` or `data/consumer_protection_act_2019.pdf`, and inspected adjacent page text where a proposed answer appeared cut off.
- Checked whether the answer responded to the question, whether a supplied reference matched the provision visible in the source, whether answer text was incomplete or visibly malformed, and whether the batch had duplicate/template warnings.
- The declared source passage was found on the declared PDF page for 50 of 50 candidates. Proposed answer text appeared in its supplied passage for 50 of 50. These are provenance observations only.

## What remains unverified

- No lawyer or qualified human reviewer validated any candidate. **All original candidates remain `PENDING_HUMAN_REVIEW`; zero were approved or legally verified.** `PROPOSED_APPROVE` means only that this first-pass screen found no clear textual defect within the asked scope.
- This report does not establish that the PDFs are the authoritative, current versions of the law. Candidate 23 specifically needs current-law checking because the source PDF itself includes a footnote saying the relevant amendment was struck down.
- No external legal sources, cases, rules, notifications, or interpretations were consulted. No legal conclusions beyond the supplied PDFs are asserted.
- PDF text extraction can contain OCR/layout artifacts. The report flags apparent fragments, but a human should visually inspect the original PDF page and its surrounding headings/clauses before making any decision.
- Duplicate/template flags are leads for review, not automatic rejection. The report did not rewrite, deduplicate, approve, or reject any original candidate.

## Recommendation meanings

- `PROPOSED_APPROVE`: textually clear for the scoped question and the supplied reference matches the source passage; still requires human review.
- `PROPOSED_REJECT`: a clear defect was visible, such as a mismatched citation, malformed question, answer that does not answer the question, or material truncation.
- `PENDING_HUMAN_REVIEW`: the text may be useful but reference, scope, uniqueness, current-law status, or interpretation needs human checking.

The recommendations exist only in this new report directory. They are not completed decisions in the Stage 14/16 review workflow and must not be imported as such.
