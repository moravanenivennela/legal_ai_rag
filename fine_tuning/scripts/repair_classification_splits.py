import json
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)

ROOT = Path("fine_tuning/dataset/expanded")
CLS = ROOT / "classification"
QA = ROOT / "legal_qa"
OUT = CLS / "repaired_v2"
OUT.mkdir(parents=True, exist_ok=True)

QA_TEST = Path("fine_tuning/dataset/qa_test.jsonl")

def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

def question_of(row):
    # Support every field used in the current files.
    for field in ("text", "question"):
        if clean(row.get(field)):
            return clean(row[field])

    raw = str(row.get("input", ""))
    match = re.search(
        r"Question:\s*(.*?)(?:\n\s*\nLegal evidence:|$)",
        raw, flags=re.S | re.I
    )
    return clean(match.group(1)) if match else clean(raw)

def label_of(row):
    value = clean(row.get("label") or row.get("source")).lower()
    if value in ("constitution", "consumer_protection", "out_of_domain"):
        return value
    return ""

def make_record(question, label, split, page=None):
    return {
        "text": question,
        "label": label,
        "source": label,
        "page": page,
        "split": split,
        "review_status": "pending",
    }

# 1. Reserve test questions FIRST so they cannot leak into train/validation.
test_records = []
seen_test = set()

for row in read_jsonl(QA_TEST):
    q = question_of(row)
    label = label_of(row)
    if q and label in ("constitution", "consumer_protection"):
        key = q.casefold()
        if key not in seen_test:
            seen_test.add(key)
            test_records.append(make_record(q, label, "test", row.get("page")))

# Add in-domain examples from the original classification test candidates.
original_test_candidates = read_jsonl(CLS / "test_candidates.jsonl")
for row in original_test_candidates:
    q = question_of(row)
    label = label_of(row)
    if q and label in ("constitution", "consumer_protection"):
        key = q.casefold()
        if key not in seen_test:
            seen_test.add(key)
            test_records.append(make_record(q, label, "test", row.get("page")))

# 2. Read QA train/validation; skip any questions reserved for test.
reserved = set(seen_test)

def qa_to_classification(rows, split):
    result = []
    for row in rows:
        q = question_of(row)
        label = label_of(row)
        key = q.casefold()
        if not q or label not in ("constitution", "consumer_protection"):
            continue
        if key in reserved:
            continue
        reserved.add(key)
        result.append(make_record(q, label, split, row.get("page")))
    return result

val_records = qa_to_classification(
    read_jsonl(QA / "validation_candidates.jsonl"), "validation"
)
train_records = qa_to_classification(
    read_jsonl(QA / "train_candidates.jsonl"), "train"
)

# 3. Collect OOD questions, deduplicating them globally.
ood_by_question = {}
for filename in (
    "train_candidates.jsonl",
    "validation_candidates.jsonl",
    "test_candidates.jsonl",
):
    for row in read_jsonl(CLS / filename):
        if label_of(row) != "out_of_domain":
            continue
        q = question_of(row)
        key = q.casefold()
        if q and key not in reserved:
            ood_by_question.setdefault(key, q)

ood_questions = list(ood_by_question.values())
random.shuffle(ood_questions)

n_train = int(len(ood_questions) * 0.70)
n_val = int(len(ood_questions) * 0.15)

ood_train = [make_record(q, "out_of_domain", "train")
             for q in ood_questions[:n_train]]
ood_val = [make_record(q, "out_of_domain", "validation")
           for q in ood_questions[n_train:n_train+n_val]]
ood_test = [make_record(q, "out_of_domain", "test")
            for q in ood_questions[n_train+n_val:]]

datasets = {
    "train": train_records + ood_train,
    "validation": val_records + ood_val,
    "test": test_records + ood_test,
}

# 4. Shuffle, save, and report distributions and overlap.
for split, rows in datasets.items():
    random.shuffle(rows)
    path = OUT / f"{split}_classification.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\n=== CLASSIFICATION DATASET REPAIR V2 ===")
question_splits = {}
for split, rows in datasets.items():
    counts = Counter(row["label"] for row in rows)
    print(f"\n{split.upper()}: {len(rows)} records")
    for label in ("constitution", "consumer_protection", "out_of_domain"):
        print(f"  {label}: {counts.get(label, 0)}")
    for row in rows:
        question_splits.setdefault(row["text"].casefold(), set()).add(split)

overlaps = sum(1 for splits in question_splits.values() if len(splits) > 1)
print(f"\nExact cross-split question overlaps: {overlaps}")
print(f"QA test questions reserved: {len(test_records)}")
print(f"Unique OOD questions allocated: {len(ood_questions)}")
print(f"Saved to: {OUT.resolve()}")
print("\nThese records remain candidates; labels still require review.")
print("Original candidate files and qa_test.jsonl were not modified.")
