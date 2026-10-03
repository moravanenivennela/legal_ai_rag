import json
import random
import re
from pathlib import Path

import pymupdf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "fine_tuning" / "dataset"

SEED = 42
random.seed(SEED)


SOURCE_FILES = {
    "constitution": DATA_DIR / "constitution_of_india.pdf",
    "consumer_protection": DATA_DIR / "consumer_protection_act_2019.pdf",
}


def repair_word_boundaries(text):
    replacements = {
        "externalaggression": "external aggression",
        "farsoin": "far so in",
        "suchsupply": "such supply",
        "jurisdictionto": "jurisdiction to",
        "constitutea": "constitute a",
        "uponthe": "upon the",
        "withdrawnat": "withdrawn at",
        "betweenparties": "between parties",
        "anysuch": "any such",
        "thefollowing": "the following",
        "inthe": "in the",
        "ofthe": "of the",
        "tothe": "to the",
        "fromthe": "from the",
        "forthe": "for the",
        "underthe": "under the",
        "withthe": "with the",
        "bythe": "by the",
        "onthe": "on the",
        "asthe": "as the",
        "suchas": "such as",
        "shallhave": "shall have",
        "maymake": "may make",
        "maybemade": "may be made",
        "shallbe": "shall be",
        "shallcease": "shall cease",
        "consistof": "consist of",
        "referredto": "referred to",
        "providedthat": "provided that",
        "subjectto": "subject to",
        "powersto": "powers to",
        "powersof": "powers of",
        "membersof": "members of",
        "Governmentshall": "Government shall",
        "Commissionwith": "Commission with",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text):
    text = text.replace("\x00", " ")

    text = repair_word_boundaries(text)

    # Remove repeated underscores used by PDF footnote separators.
    text = re.sub(r"_{5,}", " ", text)

    # Normalize non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def extract_pdf(path):
    print(f"\nReading: {path.name}")

    doc = pymupdf.open(str(path))

    pages = []

    for page_number, page in enumerate(
        tqdm(doc, desc=f"Extracting {path.name}")
    ):
        # sort=True attempts a natural top-to-bottom,
        # left-to-right reading order.
        text = page.get_text("text", sort=True)

        text = clean_text(text)

        if text:
            pages.append(
                {
                    "page": page_number + 1,
                    "text": text,
                }
            )

    doc.close()

    return pages


def split_into_chunks(text, min_words=80, max_words=220):

    # Convert line breaks to spaces after extraction.
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = []

    for sentence in sentences:
        words = sentence.split()

        if not words:
            continue

        if len(current) + len(words) <= max_words:
            current.extend(words)
        else:
            if len(current) >= min_words:
                chunks.append(" ".join(current))

            current = words

    if len(current) >= min_words:
        chunks.append(" ".join(current))

    return chunks


def build_passages():

    passages = []

    for source_name, pdf_path in SOURCE_FILES.items():

        pages = extract_pdf(pdf_path)

        for page in pages:

            chunks = split_into_chunks(
                page["text"]
            )

            for chunk_id, chunk in enumerate(chunks):

                passages.append(
                    {
                        "source": source_name,
                        "page": page["page"],
                        "chunk_id": chunk_id,
                        "context": chunk,
                    }
                )

    return passages


def save_jsonl(path, records):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def main():

    print("=" * 60)
    print("PYMUPDF LEGAL PASSAGE EXTRACTION")
    print("=" * 60)

    passages = build_passages()

    print(
        f"\nTotal passages extracted: {len(passages)}"
    )

    if not passages:
        raise RuntimeError(
            "No legal passages were extracted."
        )

    random.shuffle(passages)

    total = len(passages)

    train_end = int(total * 0.80)
    validation_end = int(total * 0.90)

    train = passages[:train_end]

    validation = passages[
        train_end:validation_end
    ]

    test = passages[
        validation_end:
    ]

    save_jsonl(
        OUTPUT_DIR / "train_passages.jsonl",
        train
    )

    save_jsonl(
        OUTPUT_DIR / "validation_passages.jsonl",
        validation
    )

    save_jsonl(
        OUTPUT_DIR / "test_passages.jsonl",
        test
    )

    metadata = {
        "extractor": "PyMuPDF",
        "seed": SEED,
        "total_passages": total,
        "train_passages": len(train),
        "validation_passages": len(validation),
        "test_passages": len(test),
        "sources": {
            name: str(path)
            for name, path in SOURCE_FILES.items()
        },
    }

    with open(
        OUTPUT_DIR / "dataset_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nExtraction complete.")

    print(
        f"Train      : {len(train)}"
    )

    print(
        f"Validation : {len(validation)}"
    )

    print(
        f"Test       : {len(test)}"
    )


if __name__ == "__main__":
    main()
