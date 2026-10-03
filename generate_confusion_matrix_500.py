import csv
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_engine import LegalRAGEngine
from evaluate_rag_answers_500 import TESTS

OUT = Path("outputs/evaluation_500_new/confusion_matrix")
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_LABELS = {
    "constitution_of_india.pdf": "constitution",
    "consumer_protection_act_2019.pdf": "consumer_protection",
}

ACTUAL_CLASSES = ["constitution", "consumer_protection"]
PREDICTED_CLASSES = [
    "constitution",
    "consumer_protection",
    "out_of_domain",
]

engine = LegalRAGEngine()
counts = Counter()
rows = []

print(f"Evaluating {len(TESTS)} questions...")

for i, (question, expected_source) in enumerate(TESTS, 1):
    actual = SOURCE_LABELS.get(expected_source)

    if actual is None:
        raise ValueError(f"Unknown expected source: {expected_source}")

    _, _, _, prediction = engine.retrieve(question)
    prediction = str(prediction).strip().lower()

    if prediction not in PREDICTED_CLASSES:
        raise ValueError(
            f"Unexpected prediction for question {i}: {prediction}"
        )

    counts[(actual, prediction)] += 1

    rows.append({
        "question_number": i,
        "question": question,
        "actual_domain": actual,
        "predicted_domain": prediction,
        "correct": actual == prediction,
    })

with open(OUT / "predictions_500.csv", "w",
          newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

matrix = [
    [counts[(actual, predicted)] for predicted in PREDICTED_CLASSES]
    for actual in ACTUAL_CLASSES
]

correct = sum(
    counts[(label, label)] for label in ACTUAL_CLASSES
)
accuracy = correct / len(TESTS) * 100

with open(OUT / "confusion_matrix_500.csv", "w",
          newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Actual / Predicted"] + PREDICTED_CLASSES)
    for label, values in zip(ACTUAL_CLASSES, matrix):
        writer.writerow([label] + values)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(matrix, cmap="Blues")

ax.set_xticks(range(len(PREDICTED_CLASSES)))
ax.set_xticklabels(PREDICTED_CLASSES, rotation=20, ha="right")
ax.set_yticks(range(len(ACTUAL_CLASSES)))
ax.set_yticklabels(ACTUAL_CLASSES)

ax.set_xlabel("Predicted Domain")
ax.set_ylabel("Actual Domain")
ax.set_title("500-Question Domain Classification Confusion Matrix")

for i, row in enumerate(matrix):
    for j, value in enumerate(row):
        ax.text(j, i, str(value), ha="center", va="center")

fig.colorbar(im, ax=ax, label="Number of Questions")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix_500.png", dpi=300)
plt.close(fig)

print("\nConfusion matrix (actual rows, predicted columns):")
print("Predicted:", PREDICTED_CLASSES)
for label, values in zip(ACTUAL_CLASSES, matrix):
    print(label, values)

print(f"\nCorrect predictions: {correct}/{len(TESTS)}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Files saved in: {OUT}")
