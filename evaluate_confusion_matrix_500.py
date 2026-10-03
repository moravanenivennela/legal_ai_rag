import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

from rag_engine import LegalRAGEngine
from evaluate_rag_answers_500 import TESTS

OUTPUT_DIR = Path("outputs/evaluation_500_new/confusion_matrix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["constitution", "consumer_protection"]
SOURCE_TO_LABEL = {
    "constitution_of_india.pdf": "constitution",
    "consumer_protection_act_2019.pdf": "consumer_protection",
}

engine = LegalRAGEngine()
y_true = []
y_pred = []
rows = []

print(f"Running domain predictions for {len(TESTS)} questions.")

for i, (question, expected_source) in enumerate(TESTS, start=1):
    expected = SOURCE_TO_LABEL.get(expected_source)

    if expected is None:
        raise ValueError(f"Unexpected source label: {expected_source}")

    try:
        _, _, _, predicted = engine.retrieve(question)
        predicted = str(predicted).strip().lower()
        correct = predicted == expected

        y_true.append(expected)
        y_pred.append(predicted)

        rows.append({
            "id": i,
            "question": question,
            "expected_domain": expected,
            "predicted_domain": predicted,
            "correct": correct,
            "error": "",
        })

        print(
            f"[{i:03d}/{len(TESTS)}] "
            f"expected={expected} predicted={predicted} "
            f"correct={correct}"
        )

    except Exception as exc:
        rows.append({
            "id": i,
            "question": question,
            "expected_domain": expected,
            "predicted_domain": "ERROR",
            "correct": False,
            "error": str(exc),
        })
        print(f"[{i:03d}/{len(TESTS)}] ERROR: {exc}")

# Include out-of-domain predictions as a separate prediction column,
# even though this particular test set has no out-of-domain true labels.
matrix_labels = ["constitution", "consumer_protection", "out_of_domain"]
cm = confusion_matrix(y_true, y_pred, labels=matrix_labels)

with (OUTPUT_DIR / "predictions_500.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=[
        "id", "question", "expected_domain", "predicted_domain",
        "correct", "error"
    ])
    writer.writeheader()
    writer.writerows(rows)

# Two-class scores are computed only for successful predictions whose
# predicted label is one of the two evaluated classes.
valid_pairs = [
    (r["expected_domain"], r["predicted_domain"])
    for r in rows
    if r["predicted_domain"] in LABELS
]
true_valid = [x[0] for x in valid_pairs]
pred_valid = [x[1] for x in valid_pairs]

accuracy_all = sum(
    r["expected_domain"] == r["predicted_domain"] for r in rows
) / len(rows)

with (OUTPUT_DIR / "confusion_matrix_report.txt").open(
    "w", encoding="utf-8"
) as f:
    f.write("500-QUESTION DOMAIN CLASSIFICATION EVALUATION\n")
    f.write("Dataset: synthetic candidate questions; manual review required.\n")
    f.write("Scope: Constitution vs Consumer Protection only.\n")
    f.write("No true out-of-domain questions are included.\n\n")
    f.write(f"Total questions: {len(rows)}\n")
    f.write(f"Correct domain predictions: {sum(r['correct'] for r in rows)}\n")
    f.write(f"Overall accuracy, counting errors as incorrect: {accuracy_all:.4f}\n")
    f.write(f"Errors/exceptions: {sum(bool(r['error']) for r in rows)}\n\n")
    f.write("Confusion matrix label order:\n")
    f.write(f"{matrix_labels}\n")
    f.write(str(cm))
    f.write("\n\n")
    f.write("Classification report for the two evaluated classes.\n")
    f.write("Review prediction errors before reporting results in a paper.\n\n")
    f.write(classification_report(
        true_valid, pred_valid, labels=LABELS,
        zero_division=0, digits=4
    ))

fig, ax = plt.subplots(figsize=(7, 6))
display_labels = ["Constitution", "Consumer Protection", "Out of domain"]
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Domain Classification Confusion Matrix\n500 Synthetic Candidate Questions")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix_500.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved outputs to:", OUTPUT_DIR)
print(" - predictions_500.csv")
print(" - confusion_matrix_report.txt")
print(" - confusion_matrix_500.png")
print("\nImportant: the test set has no true out-of-domain examples.")
