"""
Finds pairs of training examples that are semantically very similar
(high cosine similarity) but have DIFFERENT labels — these contradictions
are usually what caps classifier accuracy. Run this before adding more
data blindly.
"""
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import importlib.util

spec = importlib.util.spec_from_file_location("train_module", "train_expanded_classifier.py")
train_module = importlib.util.module_from_spec(spec)
import sys
sys.modules["train_module"] = train_module
# Only pull TRAINING_DATA without re-running the whole training script
with open("train_expanded_classifier.py", encoding="utf-8") as f:
    source = f.read()
namespace = {}
# Execute only up to where TRAINING_DATA is defined by finding the list literal
start = source.index("TRAINING_DATA = [")
end = source.index("]\n", start) + 2
exec(source[start:end], namespace)
TRAINING_DATA = namespace["TRAINING_DATA"]

texts = [t[0] for t in TRAINING_DATA]
labels = [t[1] for t in TRAINING_DATA]

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
X = np.array(embedder.embed_documents(texts))

norms = np.linalg.norm(X, axis=1, keepdims=True)
X_normed = X / norms
sim_matrix = X_normed @ X_normed.T

print("Potential label conflicts (similarity > 0.75, different labels):\n")
conflicts_found = 0
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        if labels[i] != labels[j] and sim_matrix[i, j] > 0.75:
            conflicts_found += 1
            print(f"Similarity: {sim_matrix[i,j]:.3f}")
            print(f"  [{labels[i]}]  {texts[i]}")
            print(f"  [{labels[j]}]  {texts[j]}")
            print()

if conflicts_found == 0:
    print("No high-similarity conflicts found — accuracy ceiling is likely just data volume, not label noise.")
else:
    print(f"\nTotal conflicts found: {conflicts_found}")
    print("Fix these by either: (1) relabeling one of each pair consistently,")
    print("or (2) removing the more ambiguous one entirely.")
