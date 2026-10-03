import csv
import json
from pathlib import Path

csv_path = Path("fine_tuning/dataset/qa_review_train_validation.csv")
decisions_path = Path("fine_tuning/dataset/qa_review_decisions.json")

if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

if decisions_path.exists():
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
else:
    decisions = {}

def save_decisions():
    temp_path = decisions_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    temp_path.replace(decisions_path)

pending = [
    i for i in range(len(rows))
    if str(i) not in decisions
]

print(f"\nTotal rows: {len(rows)}")
print(f"Already reviewed: {len(decisions)}")
print(f"Remaining: {len(pending)}")
print(f"Decisions are saved separately to: {decisions_path}")
print("Commands: y=approve, n=reject, s=skip, q=quit\n")

for position, i in enumerate(pending, 1):
    r = rows[i]
    print("\n" + "=" * 75)
    print(f"Question {i + 1} | Pending review {position}/{len(pending)}")
    print(f"Split: {r.get('split')} | Source: {r.get('source')} | Page: {r.get('page')}")
    print("\nQUESTION:\n", r.get("question", ""))
    print("\nANSWER:\n", r.get("answer", ""))
    print("\nLEGAL REFERENCE:\n", r.get("legal_reference", ""))
    print("\nSUPPORTING PASSAGE:\n", r.get("supporting_passage", ""))
    print("\nSOURCE PAGE TEXT:\n", (r.get("source_page_text") or "")[:1800])

    while True:
        choice = input("\nDecision [y/n/s/q]: ").strip().lower()
        if choice in {"y", "n", "s", "q"}:
            break
        print("Enter y, n, s, or q.")

    if choice == "q":
        break
    if choice == "s":
        continue

    note = ""
    if choice == "n":
        note = input("Reason for rejection (optional): ").strip()

    decisions[str(i)] = {
        "decision": "yes" if choice == "y" else "no",
        "review_notes": note
    }
    save_decisions()
    print(f"Saved decision. Total saved: {len(decisions)}")

print("\n===== PROGRESS =====")
print("Approved:", sum(d["decision"] == "yes" for d in decisions.values()))
print("Rejected:", sum(d["decision"] == "no" for d in decisions.values()))
print("Pending:", len(rows) - len(decisions))
print("Original CSV was not modified.")
print("Decision file:", decisions_path)
