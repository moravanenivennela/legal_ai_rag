import csv
import shutil
from pathlib import Path
from datetime import datetime

csv_path = Path("fine_tuning/dataset/qa_review_train_validation.csv")
backup_path = csv_path.with_name(
    f"qa_review_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

shutil.copy2(csv_path, backup_path)

with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

if not fields or "approved" not in fields:
    raise SystemExit("Missing approved column in CSV.")

def save_rows():
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

pending = [
    i for i, row in enumerate(rows)
    if (row.get("approved") or "").strip().lower() not in {"yes", "no"}
]

print("\n===== GIT BASH QA REVIEW =====")
print(f"Total rows: {len(rows)}")
print(f"Already reviewed: {len(rows) - len(pending)}")
print(f"Remaining to review: {len(pending)}")
print(f"Backup created: {backup_path}")
print("\nCheck every answer against its supporting passage and source page.")
print("Commands: y=approve, n=reject, s=skip, q=quit and save.\n")

approved_count = sum(
    (r.get("approved") or "").strip().lower() == "yes" for r in rows
)
rejected_count = sum(
    (r.get("approved") or "").strip().lower() == "no" for r in rows
)

for position, idx in enumerate(pending, start=1):
    row = rows[idx]

    print("\n" + "=" * 78)
    print(f"REVIEW {position}/{len(pending)} | CSV row {idx + 2}")
    print(f"Split: {row.get('split', '')}")
    print(f"Source: {row.get('source', '')} | Page: {row.get('page', '')}")
    print(f"\nQUESTION:\n{row.get('question', '')}")
    print(f"\nGENERATED ANSWER:\n{row.get('answer', '')}")
    print(f"\nLEGAL REFERENCE:\n{row.get('legal_reference', '')}")
    print(f"\nSUPPORTING PASSAGE:\n{row.get('supporting_passage', '')}")
    
    page_text = (row.get("source_page_text") or "").strip()
    print("\nSOURCE PAGE TEXT (first 1800 characters):")
    print(page_text[:1800] if page_text else "[No page text available]")
    if len(page_text) > 1800:
        print("[Page text truncated for display]")

    while True:
        choice = input("\nDecision [y/n/s/q]: ").strip().lower()
        if choice in {"y", "n", "s", "q"}:
            break
        print("Please enter y, n, s, or q.")

    if choice == "q":
        save_rows()
        print("\nProgress saved. You can run this script again to continue.")
        break

    if choice == "s":
        continue

    if choice == "y":
        row["approved"] = "yes"
        approved_count += 1
        row["review_notes"] = (
            row.get("review_notes") or ""
        ).strip()
    else:
        row["approved"] = "no"
        rejected_count += 1
        note = input("Reason for rejection (optional): ").strip()
        if note:
            row["review_notes"] = note

    save_rows()
    print(f"Saved. Approved: {approved_count} | Rejected: {rejected_count}")

save_rows()

print("\n===== REVIEW STATUS =====")
print(f"Approved: {sum((r.get('approved') or '').strip().lower() == 'yes' for r in rows)}")
print(f"Rejected: {sum((r.get('approved') or '').strip().lower() == 'no' for r in rows)}")
print(f"Pending: {sum((r.get('approved') or '').strip().lower() not in {'yes', 'no'} for r in rows)}")
print(f"Updated CSV: {csv_path}")
print(f"Backup CSV: {backup_path}")
