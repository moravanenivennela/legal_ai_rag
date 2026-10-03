import ast
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path.cwd()
NEW_FILE = ROOT / "fine_tuning/dataset/expanded/evaluation_500_new/legal_evaluation_500_candidates.csv"
OUT_DIR = ROOT / "outputs/evaluation_500_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()

def extract_questions(obj):
    found = []
    if isinstance(obj, dict):
        for key in ("Question", "question", "query", "text", "instruction", "prompt", "input"):
            value = obj.get(key)
            if isinstance(value, str) and len(value.strip()) > 10:
                found.append(value)
        # Common instruction/input training format
        instruction = obj.get("instruction", "")
        inp = obj.get("input", "")
        if isinstance(instruction, str) and isinstance(inp, str) and inp.strip():
            found.append(instruction + " " + inp)
    elif isinstance(obj, list):
        for item in obj:
            found.extend(extract_questions(item))
    return found

with NEW_FILE.open(encoding="utf-8-sig", newline="") as f:
    new_rows = list(csv.DictReader(f))

new_questions = {
    normalize(row.get("Question", "")): row
    for row in new_rows if normalize(row.get("Question", ""))
}

matches = []
scanned_files = 0

# Check existing datasets, excluding this new evaluation folder.
dataset_root = ROOT / "fine_tuning" / "dataset"
for path in dataset_root.rglob("*"):
    if not path.is_file():
        continue
    if "evaluation_500_new" in path.parts:
        continue

    questions = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        questions.extend(extract_questions(json.loads(line)))
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    questions.extend(extract_questions(row))
        else:
            continue
    except Exception as e:
        print(f"Skipped unreadable file: {path.relative_to(ROOT)} ({e})")
        continue

    if not questions:
        continue

    scanned_files += 1
    for question in questions:
        key = normalize(question)
        if key in new_questions:
            matches.append({
                "new_id": new_questions[key].get("ID", ""),
                "question": new_questions[key].get("Question", ""),
                "matched_file": str(path.relative_to(ROOT)),
                "matched_question": question,
            })

# Also compare against the existing hard-coded 30-question RAG list.
rag_file = ROOT / "evaluate_rag_answers.py"
if rag_file.exists():
    try:
        tree = ast.parse(rag_file.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TESTS" for t in node.targets
            ):
                tests = ast.literal_eval(node.value)
                for question, expected_source in tests:
                    key = normalize(question)
                    if key in new_questions:
                        matches.append({
                            "new_id": new_questions[key].get("ID", ""),
                            "question": new_questions[key].get("Question", ""),
                            "matched_file": "evaluate_rag_answers.py (existing RAG test)",
                            "matched_question": question,
                        })
    except Exception as e:
        print("Could not parse existing RAG test list:", e)

out_csv = OUT_DIR / "question_overlap_audit.csv"
with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
    fields = ["new_id", "question", "matched_file", "matched_question"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(matches)

print("=" * 60)
print("QUESTION OVERLAP AUDIT")
print("=" * 60)
print("New evaluation questions:", len(new_rows))
print("Existing dataset files scanned:", scanned_files)
print("Exact normalized overlap matches:", len(matches))
print("Audit file:", out_csv)

if matches:
    print("\nFirst 20 overlap matches:")
    for match in matches[:20]:
        print("-", match["new_id"], "|", match["matched_file"], "|", match["question"])
else:
    print("No exact overlap found in the scanned datasets or existing 30-question list.")

print("\nIMPORTANT:")
print("No exact overlap does not prove there is no semantic overlap or leakage.")
print("Review labels, sources, paraphrases, and train/validation/test separation.")
