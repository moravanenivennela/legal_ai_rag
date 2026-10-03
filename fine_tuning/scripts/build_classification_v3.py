import json
import shutil
from pathlib import Path

BASE = Path("fine_tuning/dataset/expanded/classification")
V2 = BASE / "repaired_v2"
CANDIDATES = BASE / "realistic_ood_candidates.jsonl"
OUT = BASE / "repaired_v3"

OUT.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def question_key(row):
    q = row.get("text") or row.get("question") or row.get("input") or ""
    return " ".join(str(q).lower().split())

train = load_jsonl(V2 / "train_classification.jsonl")
validation = load_jsonl(V2 / "validation_classification.jsonl")
test = load_jsonl(V2 / "test_classification.jsonl")
candidates = load_jsonl(CANDIDATES)

seen = {question_key(row) for row in train}
added = 0

for row in candidates:
    key = question_key(row)
    if not key or key in seen:
        continue

    # Candidate questions are added to TRAIN only.
    row["split"] = "train"
    row["review_status"] = "synthetic_candidate_not_human_verified"
    train.append(row)
    seen.add(key)
    added += 1

def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

save_jsonl(OUT / "train_classification.jsonl", train)
save_jsonl(OUT / "validation_classification.jsonl", validation)
save_jsonl(OUT / "test_classification.jsonl", test)

print("CLASSIFICATION V3 CREATED")
print(f"Training records: {len(train)}")
print(f"New OOD candidates added: {added}")
print(f"Validation records copied unchanged: {len(validation)}")
print(f"Test records copied unchanged: {len(test)}")
print(f"Output folder: {OUT}")
print("Original V2 files and blind evaluation were not modified.")
print("New OOD candidates remain marked as not human-verified.")
