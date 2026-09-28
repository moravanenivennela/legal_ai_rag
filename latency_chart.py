"""
End-to-end response latency chart: measures wall-clock time per query through
your full pipeline (retrieval + guardrail + LLM generation).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from rag_engine import LegalRAGEngine

TEST_QUERIES = [
    "What are the six consumer rights under the Consumer Protection Act?",
    "What does Article 21 of the Constitution protect?",
    "What is an unfair trade practice under consumer law?",
    "Explain the writ jurisdiction under Article 32.",
    "What is the procedure for filing a consumer complaint?",
    "What is the significance of the Preamble?",
    "What is meant by deficiency in service?",
    "What is Article 19 and what freedoms does it protect?",
    "What is the jurisdiction of the National Commission?",
    "How is the Supreme Court's jurisdiction defined?",
]

engine = LegalRAGEngine(model_name="llama3.2:1b")

latencies = []
for q in TEST_QUERIES:
    start = time.perf_counter()
    engine.retrieve(q)
    elapsed = time.perf_counter() - start
    latencies.append(elapsed)
    print(f"{elapsed:.2f}s - {q[:60]}")

latencies = np.array(latencies)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(range(1, len(latencies) + 1), latencies, color="#3b82f6")
ax.axhline(latencies.mean(), color="#ef4444", linestyle="--",
           label=f"Mean: {latencies.mean():.2f}s")
ax.set_xlabel("Query #")
ax.set_ylabel("Response Time (s)")
ax.set_title("End-to-End Query Latency", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/latency_chart.png", dpi=150)
plt.close()

print(f"\nMean latency: {latencies.mean():.2f}s | Std: {latencies.std():.2f}s")
print("Saved: outputs/latency_chart.png")
