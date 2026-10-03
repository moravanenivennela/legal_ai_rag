import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(
    "fine_tuning/dataset/expanded/classification/repaired_v2"
)
FILES = {
    "train": ROOT / "train_classification.jsonl",
    "validation": ROOT / "validation_classification.jsonl",
    "test": ROOT / "test_classification.jsonl",
}
VALID_LABELS = {
    "constitution",
    "consumer_protection",
    "out_of_domain",
}

def get_question(row):
    return str(
        row.get("text")
        or row.get("question")
        or row.get("input")
        or ""
    ).strip()

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

all_questions = {}
template_counts = Counter()
problems = []

for split, path in FILES.items():
    seen = set()
    counts = Counter()
    rows = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(
                    f"{split}:{line_number}: invalid JSON"
                )
                continue

            rows.append(row)
            label = row.get("label")
            question = get_question(row)
            normalized = normalize(question)
            counts[label] += 1

            if label not in VALID_LABELS:
                problems.append(
                    f"{split}:{line_number}: invalid label {label!r}"
                )
            if len(question) < 12:
                problems.append(
                    f"{split}:{line_number}: missing/short question"
                )
            if normalized in seen:
                problems.append(
                    f"{split}:{line_number}: duplicate question in split"
                )
            seen.add(normalized)

            if normalized in all_questions:
                problems.append(
                    f"{split}:{line_number}: exact overlap with "
                    f"{all_questions[normalized]}"
                )
            else:
                all_questions[normalized] = f"{split}:{line_number}"

            if label == "out_of_domain":
                words = re.findall(r"[a-z]+", normalized)
                if words:
                    template = " ".join(words[:4])
                    template_counts[template] += 1

    print(f"\n{split.upper()}: {len(rows)} records")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

print("\nMOST COMMON OOD OPENING PHRASES")
for phrase, count in template_counts.most_common(15):
    print(f"  {count:>3}  {phrase}")

print(f"\nAUDIT ISSUES: {len(problems)}")
for problem in problems[:50]:
    print(" -", problem)
if len(problems) > 50:
    print(f" ... plus {len(problems) - 50} more")
print("\nAudit complete. No files were modified.")
