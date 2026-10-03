import json
import random
from pathlib import Path

from generate_250_dataset import (
    PDFS,
    extract_passages,
    generate_qa,
    write_jsonl,
)

random.seed(42)

OUT_DIR = Path("fine_tuning/dataset")
PARTIAL_FILE = OUT_DIR / "qa_partial.jsonl"
TARGETS = {
    "constitution": 125,
    "consumer_protection": 125,
}

def load_jsonl(path):
    records = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        print("Skipping an invalid JSON line.")
    return records

def split_by_source_page(records, train_ratio=0.8, val_ratio=0.1):
    groups = {}
    for record in records:
        key = (record.get("source"), record.get("page"))
        groups.setdefault(key, []).append(record)

    keys = list(groups)
    random.shuffle(keys)
    n_train = int(len(keys) * train_ratio)
    n_val = int(len(keys) * val_ratio)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])

    train, validation, test = [], [], []
    for key, items in groups.items():
        if key in train_keys:
            train.extend(items)
        elif key in val_keys:
            validation.extend(items)
        else:
            test.extend(items)

    return train, validation, test

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(PARTIAL_FILE)

    seen = {
        (r.get("source"), r.get("question", "").strip().lower())
        for r in records
    }

    print("Existing saved QA records:", len(records))

    passages_by_source = {}
    for source, pdf_path in PDFS.items():
        passages = extract_passages(pdf_path, source)
        random.shuffle(passages)
        passages_by_source[source] = passages
        print(f"{source}: {len(passages)} passages")

    for source, target in TARGETS.items():
        current = sum(r.get("source") == source for r in records)
        print(f"\n{source}: {current}/{target} already saved")

        passages = passages_by_source[source]
        max_attempts = min(len(passages) * 3, target * 5)
        attempts = 0

        while current < target and attempts < max_attempts:
            passage = passages[attempts % len(passages)]
            attempts += 1

            generated = generate_qa(passage, count=1)
            for item in generated:
                question = item.get("question", "").strip()
                key = (source, question.lower())

                if not question or key in seen:
                    continue

                # Retain only records with a verified source quotation.
                quote = item.get("supporting_passage", "").strip()
                if not quote:
                    continue

                records.append(item)
                seen.add(key)
                current += 1

                # Save immediately so an interruption loses minimal progress.
                write_jsonl(PARTIAL_FILE, records)

                if current % 5 == 0 or current == target:
                    print(
                        f"Saved {source}: {current}/{target} "
                        f"(attempts: {attempts})"
                    )

                if current >= target:
                    break

        if current < target:
            print(
                f"WARNING: {source} reached {current}/{target}. "
                "The generator may need another run."
            )

    counts = {
        source: sum(r.get("source") == source for r in records)
        for source in TARGETS
    }
    print("\n===== QA GENERATION SUMMARY =====")
    print("Total records:", len(records))
    print("By source:", counts)
    print("Partial file:", PARTIAL_FILE)

    # Create split files only when at least some records were generated.
    if records:
        train, validation, test = split_by_source_page(records)
        write_jsonl(OUT_DIR / "qa_train.jsonl", train)
        write_jsonl(OUT_DIR / "qa_validation.jsonl", validation)
        write_jsonl(OUT_DIR / "qa_test.jsonl", test)

        print(
            f"Split sizes: train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}"
        )
        print(
            "Check these records for legal correctness before training. "
            "A matching quotation does not prove the answer is correct."
        )

if __name__ == "__main__":
    main()
