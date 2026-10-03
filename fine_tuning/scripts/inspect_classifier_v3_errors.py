import json
import pickle
from pathlib import Path
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

data_file = Path(
    "fine_tuning/dataset/expanded/classification/repaired_v3/test_classification.jsonl"
)
model_file = Path("models/query_classifier_v3_candidate.pkl")

rows = []
with data_file.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

with model_file.open("rb") as f:
    model = pickle.load(f)

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectors = np.asarray(embedder.embed_documents([r["text"] for r in rows]))
predictions = model.predict(vectors)

print("\n=== V3 TEST MISCLASSIFICATIONS ===")
errors = 0
for row, pred in zip(rows, predictions):
    if row["label"] != pred:
        errors += 1
        print(f"\nExpected:   {row['label']}")
        print(f"Predicted:  {pred}")
        print(f"Question:   {row['text']}")

print(f"\nTotal errors: {errors}/{len(rows)}")

print("\n=== VALIDATION CONSUMER PROTECTION ERRORS ===")
val_file = data_file.with_name("validation_classification.jsonl")
val_rows = []
with val_file.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            val_rows.append(json.loads(line))

val_vectors = np.asarray(
    embedder.embed_documents([r["text"] for r in val_rows])
)
val_predictions = model.predict(val_vectors)

count = 0
for row, pred in zip(val_rows, val_predictions):
    if row["label"] == "consumer_protection" and pred != row["label"]:
        count += 1
        print(f"\nExpected:   {row['label']}")
        print(f"Predicted:  {pred}")
        print(f"Question:   {row['text']}")

print(f"\nValidation Consumer Protection errors: {count}")
