import json
import random
import re
from pathlib import Path

from pypdf import PdfReader
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


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf(path):
    print(f"\nReading: {path.name}")

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(
        tqdm(reader.pages, desc=f"Extracting {path.name}")
    ):
        text = page.extract_text() or ""
        text = clean_text(text)

        if text:
            pages.append(
                {
                    "page": page_number + 1,
                    "text": text,
                }
            )

    return pages


def split_into_chunks(text, min_words=80, max_words=220):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []

    for sentence in sentences:
        words = sentence.split()

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
            chunks = split_into_chunks(page["text"])

            for chunk_id, chunk in enumerate(chunks):
                passages.append(
                    {
                        "source": source_name,
                        "page": page["page"],
                        "chunk_id": chunk_id,
                        "text": chunk,
                    }
                )

    return passages


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("LEGAL DATASET PREPARATION")
    print("========================================")

    passages = build_passages()

    print(f"\nTotal extracted passages: {len(passages)}")

    if not passages:
        raise RuntimeError(
            "No text could be extracted from the legal PDFs."
        )

    # Shuffle deterministically.
    random.shuffle(passages)

    # IMPORTANT:
    # We split passages before generating QA examples.
    # This helps prevent identical source passages appearing
    # in both training and evaluation data.

    total = len(passages)

    train_end = int(total * 0.80)
    validation_end = int(total * 0.90)

    train_passages = passages[:train_end]
    validation_passages = passages[train_end:validation_end]
    test_passages = passages[validation_end:]

    def convert(records):
        return [
            {
                "source": r["source"],
                "page": r["page"],
                "chunk_id": r["chunk_id"],
                "context": r["text"],
            }
            for r in records
        ]

    save_jsonl(
        OUTPUT_DIR / "train_passages.jsonl",
        convert(train_passages),
    )

    save_jsonl(
        OUTPUT_DIR / "validation_passages.jsonl",
        convert(validation_passages),
    )

    save_jsonl(
        OUTPUT_DIR / "test_passages.jsonl",
        convert(test_passages),
    )

    # Save metadata.
    metadata = {
        "seed": SEED,
        "sources": {
            name: str(path)
            for name, path in SOURCE_FILES.items()
        },
        "total_passages": total,
        "train_passages": len(train_passages),
        "validation_passages": len(validation_passages),
        "test_passages": len(test_passages),
        "split": {
            "train": "80%",
            "validation": "10%",
            "test": "10%",
        },
    }

    with open(
        OUTPUT_DIR / "dataset_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n========================================")
    print("DATASET EXTRACTION COMPLETE")
    print("========================================")
    print(f"Train passages      : {len(train_passages)}")
    print(f"Validation passages : {len(validation_passages)}")
    print(f"Test passages       : {len(test_passages)}")

    print("\nCreated:")
    print("  train_passages.jsonl")
    print("  validation_passages.jsonl")
    print("  test_passages.jsonl")
    print("  dataset_metadata.json")


if __name__ == "__main__":
    main()
