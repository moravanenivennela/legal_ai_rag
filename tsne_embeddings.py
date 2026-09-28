"""
t-SNE plot of query embeddings, colored by domain label. Visually demonstrates
that the query classifier's separation between domains is real.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from langchain_huggingface import HuggingFaceEmbeddings

TRAINING_DATA = [
    ("What are the six consumer rights under the Consumer Protection Act?", "consumer_protection"),
    ("What is the pecuniary jurisdiction of the District Commission?", "consumer_protection"),
    ("What does Article 21 of the Constitution protect?", "constitution"),
    ("Explain the writ jurisdiction under Article 32.", "constitution"),
    ("What is the statutory penalty for driving without a license?", "out_of_domain"),
    ("How do I file an income tax return?", "out_of_domain"),
    # >> paste your full TRAINING_DATA list here from generate_evaluation_visuals.py 
]

texts = [t[0] for t in TRAINING_DATA]
labels = [t[1] for t in TRAINING_DATA]

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
X = np.array(embedder.embed_documents(texts))

perplexity = min(30, max(5, len(X) // 3))
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
X_2d = tsne.fit_transform(X)

class_names = sorted(set(labels))
colors = {"consumer_protection": "#3b82f6", "constitution": "#10b981", "out_of_domain": "#f59e0b"}

plt.figure(figsize=(8, 7))
for cls in class_names:
    idx = [i for i, l in enumerate(labels) if l == cls]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=cls, s=60,
                color=colors.get(cls, None), edgecolors="black", linewidth=0.5)

plt.title("t-SNE of Query Embeddings by Domain", fontsize=13, fontweight="bold")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/tsne_query_embeddings.png", dpi=150)
plt.close()

print("Saved: outputs/tsne_query_embeddings.png")
print("Note: with very few samples per class, t-SNE can look noisy — this "
      "looks best once you have 40+ examples per domain.")
