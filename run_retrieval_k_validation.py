import time, re, json
import numpy as np
from rag_engine import LegalRAGEngine

REFERENCE_SAMPLE = {
    "What are the six consumer rights under the Consumer Protection Act?":
        "The six consumer rights are the right to safety, right to be informed, right to choose, right to be heard, right to seek redressal, and right to consumer education.",
    "What does Article 21 of the Constitution protect?":
        "Article 21 protects the right to life and personal liberty, stating no person shall be deprived of life or personal liberty except according to procedure established by law.",
    "What is Product Liability under the 2019 Act?":
        "Product liability makes a manufacturer, product seller, or product service provider liable to compensate a consumer for harm caused by a defective product or deficient service related to the product.",
    "Explain the writ jurisdiction under Article 32.":
        "Article 32 grants the Supreme Court power to issue writs including habeas corpus, mandamus, prohibition, quo warranto, and certiorari for enforcement of Fundamental Rights.",
    "What remedies are available to a consumer?":
        "Remedies include removal of defects, replacement of goods, refund of price paid, compensation for loss or injury, discontinuation of unfair trade practices, and payment of adequate costs.",
}

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def semantic_f1(embedder, generated_answer, reference_answer):
    gen_sents = split_sentences(generated_answer)
    ref_sents = split_sentences(reference_answer)
    if not gen_sents or not ref_sents:
        return None
    gen_embs = np.array(embedder.embed_documents(gen_sents))
    ref_embs = np.array(embedder.embed_documents(ref_sents))
    def cos_sim_matrix(A, B):
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T
    sim_matrix = cos_sim_matrix(gen_embs, ref_embs)
    precision = sim_matrix.max(axis=1).mean()
    recall = sim_matrix.max(axis=0).mean()
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

engine = LegalRAGEngine(model_name="llama3.2:1b")
K_VALUES = [2, 4, 6, 8, 10]
results_by_k = {}

for k in K_VALUES:
    print(f"\n=== top_n = {k} ===")
    f1s = []
    for q, ref in REFERENCE_SAMPLE.items():
        dense_hits, _ = engine.dense_search(q, k=10)
        sparse_hits = engine.sparse_search(q, k=10)
        fused = engine.reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=max(k, 10))
        contexts = engine.rerank(q, fused, top_n=k)
        answer = ""
        for token in engine.generate_stream(q, contexts):
            answer += token
        result = semantic_f1(engine.embedding_fn, answer, ref)
        if result:
            f1s.append(result[2] * 100)
            print(f"  {q[:40]:<40} F1={result[2]*100:.1f}%")
    results_by_k[k] = float(np.mean(f1s)) if f1s else 0.0

with open("retrieval_k_validation.json", "w") as f:
    json.dump(results_by_k, f, indent=2)

print("\n\nRESULTS BY K:", results_by_k)

import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300
plt.figure(figsize=(8,5))
ks = list(results_by_k.keys())
f1_vals = list(results_by_k.values())
plt.plot(ks, f1_vals, marker='o', linewidth=2, color="#3b82f6", markersize=8)
for k, v in zip(ks, f1_vals):
    plt.annotate(f"{v:.1f}%", (k, v), textcoords="offset points", xytext=(0,8), ha='center', fontweight='bold')
plt.xlabel("Number of Retrieved/Reranked Chunks (top-k)")
plt.ylabel("F1-score (%)")
plt.title("RAG Retrieval Validation Curve: Chunk Count vs. Answer Quality", fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/rag_performance/retrieval_k_validation_curve.png")
plt.close()
print("Saved: outputs/rag_performance/retrieval_k_validation_curve.png")
