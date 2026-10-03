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


def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"_{5,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"_{5,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def repair_token(text):
    # Repair only clear English word-boundary joins found in the PDFs.
    replacements = {
        "Commissionand": "Commission and",
        "Commission,shall": "Commission, shall",
        "Commission,as": "Commission, as",
        "Commission,or": "Commission, or",
        "informationreceived": "information received",
        "thecomplaint": "the complaint",
        "includingparties": "including parties",
        "mattersrelating": "matters relating",
        "agreedin": "agreed in",
        "dutyof": "duty of",
        "ofthe": "of the",
        "inthe": "in the",
        "tothe": "to the",
        "onthe": "on the",
        "underthe": "under the",
        "forthe": "for the",
        "bythe": "by the",
        "fromthe": "from the",
        "shallbe": "shall be",
        "maybemade": "may be made",
        "referredto": "referred to",
        "subjectto": "subject to",
        "providedthat": "provided that",
        "suchas": "such as",
        "otherfacts": "other facts",
        "theDistrict": "the District",
        "Replacementof": "Replacement of",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"_{5,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def extract_page_words(page):
    text = page.get_text("text")

    if not text:
        return ""

    # Repair only clearly observed PDF text-layer word joins.
    replacements = {
        "Commissionsand": "Commissions and",
        "Commissionto": "Commission to",
        "mediatorsreferred": "mediators referred",
        "inthe": "in the",
        "ofthe": "of the",
        "tothe": "to the",
        "onthe": "on the",
        "underthe": "under the",
        "forthe": "for the",
        "bythe": "by the",
        "fromthe": "from the",
        "informationreceived": "information received",
        "thecomplaint": "the complaint",
        "includingparties": "including parties",
        "mattersrelating": "matters relating",
        "agreedin": "agreed in",
        "Commission,as": "Commission, as",
        "Commission,or": "Commission, or",
        "Commission,shall": "Commission, shall",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("\x00", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"_{5,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)

    return text.strip()

def extract_pdf(path):

    print(f"\nReading: {path.name}")

    document = pymupdf.open(str(path))

    pages = []

    for page_number, page in enumerate(
        tqdm(
            document,
            desc=f"Extracting {path.name}"
        )
    ):

        text = extract_page_words(page)

        if text:
            pages.append(
                {
                    "page": page_number + 1,
                    "text": text,
                }
            )

    document.close()

    return pages


def split_into_chunks(
    text,
    min_words=80,
    max_words=220
):

    words = text.split()

    chunks = []

    current = []

    for word in words:

        if len(current) < max_words:
            current.append(word)

        else:

            if len(current) >= min_words:
                chunks.append(
                    " ".join(current)
                )

            current = [word]

    if len(current) >= min_words:
        chunks.append(
            " ".join(current)
        )

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
    print("WORD-LEVEL LEGAL PDF EXTRACTION")
    print("=" * 60)

    passages = build_passages()

    print(
        f"\nTotal passages extracted: {len(passages)}"
    )

    if not passages:
        raise RuntimeError(
            "No passages were extracted."
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
        "extractor": "PyMuPDF word-level",
        "seed": SEED,
        "total_passages": total,
        "train_passages": len(train),
        "validation_passages": len(validation),
        "test_passages": len(test),
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
