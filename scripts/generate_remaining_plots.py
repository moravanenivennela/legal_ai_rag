import matplotlib.pyplot as plt, numpy as np, os
os.makedirs('outputs/plots', exist_ok=True)
metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
bm25 = [0.90, 0.96, 0.97, 0.93]
dense = [0.98, 0.99, 1.00, 0.98]
hybrid = [1.00, 1.00, 1.00, 1.00]
x = np.arange(len(metrics)); w = 0.25
plt.figure(figsize=(8, 5))
plt.bar(x - w, bm25, w, label='BM25 (Lexical)', color='#7293CB')
plt.bar(x, dense, w, label='BGE-M3 (Dense)', color='#E1974C')
plt.bar(x + w, hybrid, w, label='Hybrid (RRF + Reranker)', color='#84BA5B')
plt.ylabel('Score / Ratio', fontweight='bold')
plt.title('Fig 4. Retrieval Strategy Performance Comparison', fontweight='bold')
plt.xticks(x, metrics, fontweight='bold'); plt.ylim(0.8, 1.05); plt.legend(loc='lower right'); plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(); plt.savefig('outputs/plots/retrieval_performance.png', dpi=300)
print('Generated outputs/plots/retrieval_performance.png')
