import json
import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

ROOT = Path.cwd()
import sys
sys.path.insert(0, str(ROOT))
CSV_PATH = ROOT / "fine_tuning/dataset/expanded/domain_guardrail_500/domain_guardrail_500_candidate_questions.csv"
CLASSIFIER_PATH = ROOT / "query_classifier.pkl"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["constitution", "consumer_protection", "out_of_domain"]

df = pd.read_csv(CSV_PATH)

question_col = next((c for c in ["Question", "question", "text"] if c in df.columns), None)
label_col = next((c for c in ["Expected", "expected", "label", "Correct"] if c in df.columns), None)

if question_col is None or label_col is None:
    raise ValueError(f"Could not identify question/label columns. CSV columns: {list(df.columns)}")

df = df.dropna(subset=[question_col, label_col]).copy()
df[question_col] = df[question_col].astype(str).str.strip()
df[label_col] = df[label_col].astype(str).str.strip().str.lower()

if len(df) != 500:
    raise ValueError(f"Expected 500 valid questions, found {len(df)}")
if not set(df[label_col]).issubset(set(LABELS)):
    raise ValueError(f"Unexpected labels: {sorted(set(df[label_col]) - set(LABELS))}")
if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")

print("Loading existing production classifier...")
with open(CLASSIFIER_PATH, "rb") as f:
    classifier = pickle.load(f)

print("Loading BAAI/bge-m3 embedding model...")
embedder = SentenceTransformer("BAAI/bge-m3")

from rag_engine import LegalRAGEngine
engine = LegalRAGEngine.__new__(LegalRAGEngine)
engine.query_classifier = classifier

predictions = []
for i, question in enumerate(df[question_col].tolist(), start=1):
    vector = embedder.encode(question, normalize_embeddings=True)
    prediction = engine.classify_query_domain(question, vector)
    predictions.append(str(prediction).strip().lower())
    if i % 25 == 0:
        print(f"Evaluated {i}/500 questions")

df["Predicted"] = predictions
df["Correct"] = df[label_col] == df["Predicted"]

pred_path = OUTPUT_DIR / "domain_guardrail_500_candidate_predictions.csv"
df.to_csv(pred_path, index=False, encoding="utf-8-sig")

cm = confusion_matrix(df[label_col], df["Predicted"], labels=LABELS)
report = classification_report(df[label_col], df["Predicted"], labels=LABELS, output_dict=True, zero_division=0)

metrics = {
    "evaluation_type": "synthetic candidate diagnostic; not an independently verified benchmark",
    "sample_count": int(len(df)),
    "class_counts": {str(k): int(v) for k, v in df[label_col].value_counts().items()},
    "labels_order": LABELS,
    "accuracy": float(accuracy_score(df[label_col], df["Predicted"])),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "warning": "Synthetic candidates may echo routing rules. Manually review labels and near-duplicates before reporting research results."
}
with open(OUTPUT_DIR / "domain_guardrail_500_candidate_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("500 Synthetic Candidate Questions: Confusion Matrix")
plt.colorbar()
ticks = range(len(LABELS))
plt.xticks(list(ticks), LABELS, rotation=25, ha="right")
plt.yticks(list(ticks), LABELS)
for r in range(len(LABELS)):
    for c in range(len(LABELS)):
        plt.text(c, r, str(cm[r, c]), ha="center", va="center")
plt.xlabel("Predicted label")
plt.ylabel("Expected label")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "domain_guardrail_500_candidate_confusion_matrix.png", dpi=200)
plt.close()

df[df["Correct"] == False].to_csv(
    OUTPUT_DIR / "domain_guardrail_500_candidate_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== SYNTHETIC CANDIDATE DIAGNOSTIC ===")
print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
print("\nConfusion matrix (rows=expected, columns=predicted):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
print("\nClassification report:")
print(classification_report(df[label_col], df["Predicted"], labels=LABELS, zero_division=0))
print("\nSaved predictions:", pred_path)
print("Saved metrics:", OUTPUT_DIR / "domain_guardrail_500_candidate_metrics.json")
print("Saved confusion matrix:", OUTPUT_DIR / "domain_guardrail_500_candidate_confusion_matrix.png")
print("Saved errors:", OUTPUT_DIR / "domain_guardrail_500_candidate_errors.csv")
print("\nIMPORTANT: This synthetic diagnostic is not a verified independent benchmark.")
