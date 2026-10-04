import time
import matplotlib.pyplot as plt
import numpy as np
import ollama
from rag_engine import LegalRAGEngine

# Check which models are actually installed before testing
installed = [m.model for m in ollama.list().models]
print("Installed Ollama models:", installed)

MODELS_TO_TEST = [m for m in ["llama3.2:1b", "llama3.2:3b"] if any(m in i for i in installed)]
print("Testing models:", MODELS_TO_TEST)

TEST_QUESTIONS = [
    "What are the six consumer rights under the Consumer Protection Act?",
    "What is the pecuniary jurisdiction of the District Commission?",
    "What is Product Liability under the 2019 Act?",
    "What does Article 21 of the Constitution protect?",
    "Explain the writ jurisdiction under Article 32.",
    "How does Article 246 distribute legislative powers?",
    "What are the Directive Principles of State Policy?",
]

results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
    engine = LegalRAGEngine(model_name=model_name)

    accs, precs, recs, f1s, grounds, times = [], [], [], [], [], []

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

        accs.append(answer_metrics["overlap_threshold_pass"])
        precs.append(answer_metrics["precision"])
        recs.append(answer_metrics["recall"])
        f1s.append(answer_metrics["f1"])
        grounds.append(groundedness)
        times.append(elapsed)

        print(f"  Q: {q[:50]}... | groundedness={groundedness:.1f}% | f1={answer_metrics['f1']:.1f}% | time={elapsed:.2f}s")

    results[model_name] = {
        "overlap_threshold_pass": np.mean(accs) if accs else 0,
        "precision": np.mean(precs) if precs else 0,
        "recall": np.mean(recs) if recs else 0,
        "f1": np.mean(f1s) if f1s else 0,
        "groundedness": np.mean(grounds) if grounds else 0,
        "avg_time": np.mean(times) if times else 0,
    }

print(f"\n\n{'='*70}")
print("SUMMARY — RAG Lexical-Overlap Diagnostics Across LLM Models")
print(f"{'='*70}")
print(f"{'Model':<18}{'F1>=30% pass':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Groundedness':<14}{'Avg Time':<10}")
for model_name, m in results.items():
    print(f"{model_name:<18}{m['overlap_threshold_pass']:<11.1f}%{m['precision']:<11.1f}%{m['recall']:<11.1f}%{m['f1']:<11.1f}%{m['groundedness']:<13.1f}%{m['avg_time']:<9.2f}s")

# --- Comparison chart, paper Fig-13 style ---
metrics_to_plot = ["overlap_threshold_pass", "precision", "recall", "f1", "groundedness"]
metric_labels = ["F1≥30% pass", "Token precision", "Token recall", "Token F1", "NLI groundedness"]

x = np.arange(len(metrics_to_plot))
width = 0.35 if len(MODELS_TO_TEST) == 2 else 0.8 / len(MODELS_TO_TEST)

fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#3b82f6", "#c5a059", "#10b981", "#9333EA"]

for i, model_name in enumerate(MODELS_TO_TEST):
    vals = [results[model_name][m] for m in metrics_to_plot]
    offset = (i - (len(MODELS_TO_TEST)-1)/2) * width
    bars = ax.bar(x + offset, vals, width, label=model_name, color=colors[i % len(colors)])
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("RAG lexical-overlap diagnostics across LLM models\n(Threshold pass is F1≥30%, not accuracy)", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/llm_model_comparison.png", dpi=150)
plt.close()
print("\nSaved: outputs/llm_model_comparison.png")
