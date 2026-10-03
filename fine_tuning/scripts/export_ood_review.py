import csv
import json
from pathlib import Path

source = Path(
    "fine_tuning/dataset/expanded/classification/"
    "realistic_ood_candidates.jsonl"
)
output = Path(
    "fine_tuning/dataset/expanded/classification/"
    "realistic_ood_review.csv"
)

with source.open(encoding="utf-8") as f, \
     output.open("w", newline="", encoding="utf-8-sig") as out:
    writer = csv.DictWriter(
        out,
        fieldnames=[
            "text",
            "label",
            "review_status",
            "review_notes",
        ],
    )
    writer.writeheader()

    for line in f:
        row = json.loads(line)
        writer.writerow({
            "text": row.get("text", ""),
            "label": row.get("label", "out_of_domain"),
            "review_status": "pending",
            "review_notes": "",
        })

print(f"Review sheet created: {output}")
print("Open it in Excel and set review_status to approved or rejected.")
print("Do not change the original candidate JSONL file.")
