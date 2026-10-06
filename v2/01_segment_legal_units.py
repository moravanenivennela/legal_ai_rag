"""
STEP 2a - Article/Section-aware segmentation of the source PDFs.

Input : data/constitution_of_india.pdf, data/consumer_protection_act_2019.pdf
Output: data_v2/legal_units.jsonl        (one record per Article / Section)
        data_v2/segmentation_report.json (coverage + sanity checks)

Each unit keeps source, number, heading, clean text, page range, and an
article-level split (split_strict) so no unit appears in two splits.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data_v2"
OUT_DIR.mkdir(exist_ok=True)

SEED = "legal-ai-v2-seed-42"
SPLIT_RATIOS = (0.70, 0.10, 0.20)  # train / validation / test by UNIT

HEADER_PATTERNS = [
    r"^THE CONSTITUTION OF\s+INDIA\s*$",
    r"^\(Part [IVXLC]+[A-Z]*\.?\s*[—–-].*\)\s*$",
    r"^\(Appendix.*\)\s*$",
    r"^THE GAZETTE OF INDIA EXTRAORDINARY\s*$",
    r"^SEC\.\s*\d+\]\s*$",
    r"^\[PART II\s*[—–-].*$",
    r"^\d{1,3}\s*$",                      # bare page numbers
    r"^[ivxlc]+\s*$",                     # roman page numbers
]
HEADER_RE = [re.compile(p, re.I) for p in HEADER_PATTERNS]

SEPARATOR = re.compile(r"^_{5,}\s*$")
FOOTNOTE_FALLBACK = re.compile(
    r"^\d{1,3}\.\s+(Ins\.|Subs\.|Added|Omitted|Re-?numbered|Renumbered|Rep\.|"
    r"The (words|figures|brackets|expression|letters|entry|entries)|"
    r"Substituted|Inserted|Cl\.|Sub-clause|Art\.)"
)
OMITTED_RE = re.compile(r"\b(omitted|repealed)\b", re.I)


def clean_page_lines(text):
    lines = [l.rstrip() for l in text.splitlines()]
    kept = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        if any(r.match(s) for r in HEADER_RE):
            continue
        kept.append(s)
    # footnotes sit at the bottom of the page, after a ______ separator line
    cut = next((i for i, s in enumerate(kept) if SEPARATOR.match(s)), None)
    if cut is None:
        cut = next((i for i, s in enumerate(kept) if FOOTNOTE_FALLBACK.match(s)), None)
    if cut is not None:
        kept = kept[:cut]
    return kept


def normalise(s):
    s = re.sub(r"\b\d{1,3}\[", "", s)           # footnote markers like 1[
    s = s.replace("]", "")
    s = re.sub(r"\b\d{1,3}\s*\*+", " ", s)         # footnote marker + stars, e.g. 2***
    s = re.sub(r"\*(\s*\*)+", " ", s)           # *  *  * placeholders
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)      # hyphenation across lines
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_pages(pdf_path):
    doc = fitz.open(str(pdf_path))
    return [doc[i].get_text() for i in range(len(doc))]


def num_key(label):
    m = re.match(r"(\d+)([A-Z]*)", label)
    return int(m.group(1)), m.group(2)


def split_for(unit_id):
    h = int(hashlib.sha256((SEED + unit_id).encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < SPLIT_RATIOS[0]:
        return "train"
    if h < SPLIT_RATIOS[0] + SPLIT_RATIOS[1]:
        return "validation"
    return "test"


# --------------------------------------------------------------- Constitution
CAND_START = re.compile(r"(?<![\w(])(\d{1,3}[A-Z]{0,2})\.\s+(?=[A-Z])")
HEADING = re.compile(r"([A-Z][^—–]{2,170}?)\.\s*[—–]\s*")
INNER_NUM = re.compile(r"\s\d{1,3}[A-Z]{0,2}\.\s")  # heading must not contain another "N. "


def segment_constitution(pdf_path):
    pages = read_pages(pdf_path)
    body_start = next(
        (i for i, t in enumerate(pages)
         if re.search(r"1\.\s*Name and territory of the Union[^.]{0,40}\.\s*[—–]", t)), 0
    )
    stream, offsets = [], []   # concatenated text + (char_offset, page_no)
    pos = 0
    for i in range(body_start, len(pages)):
        txt = normalise(" ".join(clean_page_lines(pages[i]))) + " "
        offsets.append((pos, i + 1))
        stream.append(txt)
        pos += len(txt)
    full = "".join(stream)

    end = re.search(r"\bTHE SCHEDULES\b|\bFIRST SCHEDULE\b", full[len(full) // 2:])
    if end:
        full = full[: len(full) // 2 + end.start()]

    def page_at(off):
        p = offsets[0][1]
        for o, pg in offsets:
            if o <= off:
                p = pg
            else:
                break
        return p

    hits, last = [], (0, "")
    for m in CAND_START.finditer(full):
        label = m.group(1)
        n, suf = num_key(label)
        if n < last[0] or n > last[0] + 25:
            continue
        if n == last[0] and suf <= last[1]:
            continue
        h = HEADING.match(full, m.end())
        if not h or INNER_NUM.search(h.group(1)):
            continue
        hits.append((m.start(), h.end(), label, h.group(1).strip()))
        last = (n, suf)

    units = []
    for k, (s, e, label, heading) in enumerate(hits):
        nxt = hits[k + 1][0] if k + 1 < len(hits) else len(full)
        text = full[e:nxt].strip()
        units.append({
            "source": "constitution",
            "number": label,
            "heading": heading,
            "text": text,
            "start_page": page_at(s),
            "end_page": page_at(max(s, nxt - 1)),
        })
    return units


# ------------------------------------------------------ Consumer Protection Act
def segment_cpa(pdf_path):
    pages = read_pages(pdf_path)
    body_start = 0
    for i, t in enumerate(pages):
        if re.search(r"BE it enacted by Parliament", t, re.I):
            body_start = i
            break
    lines_by_page = []
    for i in range(body_start, len(pages)):
        for s in clean_page_lines(pages[i]):
            lines_by_page.append((i + 1, s))

    starts, expect = [], 1
    for idx, (pg, s) in enumerate(lines_by_page):
        m = re.match(r"^(\d{1,3})\.\s+(.*)$", s)
        if m and int(m.group(1)) in (expect, expect + 1):
            starts.append((idx, m.group(1)))
            expect = int(m.group(1)) + 1

    units = []
    for k, (idx, label) in enumerate(starts):
        nxt = starts[k + 1][0] if k + 1 < len(starts) else len(lines_by_page)
        chunk = lines_by_page[idx:nxt]
        raw = " ".join(s for _, s in chunk)
        raw = re.sub(r"^\d{1,3}\.\s+", "", raw)
        text = normalise(raw)
        units.append({
            "source": "consumer_protection",
            "number": label,
            "heading": "",
            "text": text,
            "start_page": chunk[0][0],
            "end_page": chunk[-1][0],
        })
    return units


def finalise(units):
    out = []
    for u in units:
        uid = f"{u['source']}:{u['number']}"
        words = len(u["text"].split())
        u.update({
            "unit_id": uid,
            "word_count": words,
            "is_omitted": bool(OMITTED_RE.search(u["text"][:160])) and words < 60,
            "split_strict": split_for(uid),
        })
        out.append(u)
    return out


def main():
    data = ROOT / "data"
    units = []
    units += segment_constitution(data / "constitution_of_india.pdf")
    units += segment_cpa(data / "consumer_protection_act_2019.pdf")
    units = finalise(units)

    with open(OUT_DIR / "legal_units.jsonl", "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    rep = {"total_units": len(units)}
    for src in ("constitution", "consumer_protection"):
        us = [u for u in units if u["source"] == src]
        nums = [num_key(u["number"])[0] for u in us]
        expected = range(1, (395 if src == "constitution" else 107) + 1)
        missing = sorted(set(expected) - set(nums))
        usable = [u for u in us if not u["is_omitted"] and u["word_count"] >= 15]
        rep[src] = {
            "units": len(us),
            "usable_units(>=15 words, not omitted)": len(usable),
            "omitted_or_repealed": sum(u["is_omitted"] for u in us),
            "missing_numbers_vs_expected_range": missing[:80],
            "n_missing": len(missing),
            "median_words": sorted(u["word_count"] for u in us)[len(us) // 2] if us else 0,
            "max_words": max((u["word_count"] for u in us), default=0),
            "splits": {s: sum(u["split_strict"] == s for u in us) for s in ("train", "validation", "test")},
        }
    (OUT_DIR / "segmentation_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
