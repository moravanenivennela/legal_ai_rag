"""
Identifies exactly which training examples the classifier gets wrong under
cross-validation, so we can inspect and fix/relabel/remove them individually
instead of guessing.
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from langchain_huggingface import HuggingFaceEmbeddings

# Import the same TRAINING_DATA from your training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_expanded_classifier", "train_expanded_classifier.py")
mod = importlib.util.module_from_spec(spec)
import sys
sys.argv = ["dummy"]  # prevent the script's own main code from re-running side effects if any

# Safer: just re-read the TRAINING_DATA list directly by exec'ing only that part
with open("train_expanded_classifier.py") as f:
    source = f.read()

namespace = {}
# Extract just the TRAINING_DATA list definition and execute it in isolation
start = source.index("TRAINING_DATA = [")
end = source.index("]", source.index("out_of_domain HARD NEGATIVES")) 
# fallback: just exec everything up to the embedder line safely
exec(source[start:source.index("print(f\"Total training examples")], namespace)
TRAINING_DATA = namespace["TRAINING_DATA"]

texts = [t[0] for t in TRAINING_DATA]
labels = [t[1] for t in TRAINING_DATA]

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
X = np.array(embedder.embed_documents(texts))
y = np.array(labels)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000, C=1.0)
y_pred = cross_val_predict(clf, X, y, cv=cv)

print("MISCLASSIFIED EXAMPLES:")
print("=" * 80)
n_wrong = 0
for text, true_label, pred_label in zip(texts, y, y_pred):
    if true_label != pred_label:
        n_wrong += 1
        print(f"TRUE: {true_label:<20} PRED: {pred_label:<20} | {text}")

print("=" * 80)
print(f"Total misclassified: {n_wrong} / {len(texts)}")
