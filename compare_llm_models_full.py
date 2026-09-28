import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import ollama
from rag_engine import LegalRAGEngine

GROUNDEDNESS_PASS_THRESHOLD = 70.0

installed = [m.model for m in ollama.list().models]
print("Installed Ollama models:", installed)

MODELS_TO_TEST = [m for m in ["llama3.2:1b", "llama3.2:3b"] if any(m in i for i in installed)]
if not MODELS_TO_TEST:
    MODELS_TO_TEST = installed[:2]
print("Testing models:", MODELS_TO_TEST)

TEST_QUESTIONS = [
    "What are the six consumer rights under the Consumer Protection Act?",
    "What is the pecuniary jurisdiction of the District Commission?",
    "What is Product Liability under the 2019 Act?",
    "What is an unfair trade practice under consumer law?",
    "What is a misleading advertisement?",
    "What are the powers of the State Commission?",
    "What remedies are available to a consumer?",
    "What is meant by deficiency in service?",
    "What does Article 21 of the Constitution protect?",
    "Explain the writ jurisdiction under Article 32.",
    "How does Article 246 distribute legislative powers?",
    "What are the Directive Principles of State Policy?",
    "What is the significance of the Preamble?",
    "What is Article 14 about equality before law?",
    "What powers does the President have under the Constitution?",
    "What is Article 19 and what freedoms does it protect?",
    "How is the Supreme Court's jurisdiction defined?",
    "What is the procedure to amend the Constitution?",
    "What are Fundamental Rights under Part III?",
    "What is the role of the Governor under the Constitution?",
]

results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
    engine = LegalRAGEngine(model_name=model_name)

    accs, precs, recs, f1s, grounds, times = [], [], [], [], [], []
    binary_labels = []

    for q in TEST_QUESTIONS:
        start = time.time()
        contexts, is_confident, min_distance, predicted_domain = engine.retrieve(q)

        if not is_confident:
            continue

        full_answer = ""
        for token in engine.generate_stream(q, contexts):
            full_answer += token

        elapsed = time.time() - start
        groundedness = engine.check_groundedness(full_answer, contexts)
        answer_metrics = engine.compute_answer_metrics(full_answer, contexts)

        accs.append(answer_metrics["accuracy"])
        precs.append(answer_metrics["precision"])
        recs.append(answer_metrics["recall"])
        f1s.append(answer_metrics["f1"])
        grounds.append(groundedness)
        times.append(elapsed)
        binary_labels.append(1 if groundedness >= GROUNDEDNESS_PASS_THRESHOLD else 0)

        print(f"  Q: {q[:55]:<55} | grounded={groundedness:5.1f}% | f1={answer_metrics['f1']:5.1f}% | t={elapsed:.2f}s")

    binary_labels = np.array(binary_labels)
    TP = int(np.sum(binary_labels == 1))
    FN = int(np.sum(binary_labels == 0))
    total = len(binary_labels)
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    results[model_name] = {
        "accuracy": np.mean(accs) if accs else 0,
        "precision": np.mean(precs) if precs else 0,
        "recall": np.mean(recs) if recs else 0,
        "f1": np.mean(f1s) if f1s else 0,
        "groundedness": np.mean(grounds) if grounds else 0,
        "sensitivity_proxy": sens * 100,
        "avg_time": np.mean(times) if times else 0,
        "n_tested": total,
    }

print(f"\n\n{'='*70}")
print("SUMMARY — RAG Performance Across LLM Models")
print(f"{'='*70}")
header = f"{'Model':<16}{'Accuracy':<11}{'Precision':<11}{'Recall':<11}{'F1':<11}{'Groundedness':<14}{'AvgTime':<9}"
print(header)
for model_name, m in results.items():
    print(f"{model_name:<16}{m['accuracy']:<10.1f}%{m['precision']:<10.1f}%{m['recall']:<10.1f}%"
          f"{m['f1']:<10.1f}%{m['groundedness']:<13.1f}%{m['avg_time']:<8.2f}s")

metrics_to_plot = ["accuracy", "precision", "recall", "f1", "groundedness"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score", "Groundedness"]

x = np.arange(len(metrics_to_plot))
width = 0.35 if len(MODELS_TO_TEST) == 2 else 0.8 / max(len(MODELS_TO_TEST), 1)

fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#3b82f6", "#c5a059", "#10b981", "#9333EA"]

for i, model_name in enumerate(MODELS_TO_TEST):
    vals = [results[model_name][m] for m in metrics_to_plot]
    offset = (i - (len(MODELS_TO_TEST) - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width, label=model_name, color=colors[i % len(colors)])
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("RAG Answer Quality Across Different LLM Models\n(Same hybrid retrieval pipeline, different generation models)",
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/llm_model_comparison_full.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
model_names = list(results.keys())
avg_times = [results[m]["avg_time"] for m in model_names]
bars = ax.bar(model_names, avg_times, color=colors[:len(model_names)])
for bar, val in zip(bars, avg_times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f}s",
            ha="center", fontweight="bold")
ax.set_ylabel("Avg Response Time (s)")
ax.set_title("End-to-End Response Time by Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/llm_model_response_time.png", dpi=150)
plt.close()

print("\nSaved: outputs/llm_model_comparison_full.png")
print("Saved: outputs/llm_model_response_time.png")

with open("outputs/llm_comparison_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Saved: outputs/llm_comparison_results.pkl")
