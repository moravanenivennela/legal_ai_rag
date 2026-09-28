import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

with open("llm_results_incremental.json") as f:
    results = json.load(f)

models = list(results.keys())
print("Models found in results file:", models)

metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
colors = ["#3b82f6", "#c5a059", "#10b981", "#9333EA"]

# ============================================================
# 1. Individual metric bar charts
# ============================================================
for metric_key, metric_label in zip(metrics_to_plot, metric_labels):
    plt.figure(figsize=(8, 5))
    vals = [results[m][metric_key] for m in models]
    bars = plt.bar(models, vals, color="#3b82f6")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontweight="bold")
    plt.ylabel(f"{metric_label} (%)")
    plt.ylim(0, 105)
    plt.title(f"LLM {metric_label} Comparison Through RAG Pipeline (n=30 questions)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/llm_performance/llm_{metric_key}_comparison.png")
    plt.close()
    print(f"Saved: llm_{metric_key}_comparison.png")

# ============================================================
# 2. Latency comparison
# ============================================================
plt.figure(figsize=(8, 5))
times_v = [results[m]["avg_time"] for m in models]
bars = plt.bar(models, times_v, color="#c5a059")
for bar, val in zip(bars, times_v):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}s", ha="center", fontweight="bold")
plt.ylabel("Avg End-to-End Response Latency (s)")
plt.title("LLM Response Latency Comparison (Retrieval + Generation)", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/llm_performance/llm_latency_comparison.png")
plt.close()
print("Saved: llm_latency_comparison.png")

# ============================================================
# 3. Groundedness (answer faithfulness) comparison
# ============================================================
plt.figure(figsize=(8, 5))
ground_v = [results[m]["groundedness"] for m in models]
bars = plt.bar(models, ground_v, color="#ef4444")
for bar, val in zip(bars, ground_v):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontweight="bold")
plt.ylabel("Groundedness / Faithfulness Score (%)")
plt.ylim(0, 105)
plt.title("Answer Faithfulness (NLI-based Groundedness) Across LLM Models", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/rag_performance/answer_faithfulness_comparison.png")
plt.close()
print("Saved: answer_faithfulness_comparison.png")

# ============================================================
# 4. Grouped bar: all 4 metrics together
# ============================================================
x = np.arange(len(metrics_to_plot))
n_models = len(models)
width = 0.8 / max(n_models, 1)
fig, ax = plt.subplots(figsize=(11, 6))
for i, model_name in enumerate(models):
    vals = [results[model_name][m] for m in metrics_to_plot]
    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width, label=model_name, color=colors[i % len(colors)])
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 2),
                    textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 110)
ax.set_title("LLM Answer Quality Comparison Through Hybrid RAG Pipeline (n=30)", fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/llm_performance/llm_grouped_metrics_comparison.png")
plt.close()
print("Saved: llm_grouped_metrics_comparison.png")

# ============================================================
# 5. Speed vs Quality scatter
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
for i, model_name in enumerate(models):
    ax.scatter(results[model_name]["avg_time"], results[model_name]["f1"], s=200,
               color=colors[i % len(colors)], edgecolor="black", zorder=3)
    ax.annotate(model_name, (results[model_name]["avg_time"], results[model_name]["f1"]),
                xytext=(8, 8), textcoords="offset points", fontweight="bold")
ax.set_xlabel("Avg Response Time (s)")
ax.set_ylabel("F1-score (%)")
ax.set_title("Speed vs. Quality Tradeoff Across LLM Models", fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/llm_performance/llm_speed_vs_quality.png")
plt.close()
print("Saved: llm_speed_vs_quality.png")

# ============================================================
# 6. Radar chart
# ============================================================
categories = metric_labels
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, model_name in enumerate(models):
    vals = [results[model_name][m] for m in metrics_to_plot]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, label=model_name, color=colors[i % len(colors)])
    ax.fill(angles, vals, alpha=0.05, color=colors[i % len(colors)])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 100)
ax.set_title("LLM Performance Radar Comparison", fontweight="bold", pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("outputs/llm_performance/llm_radar_comparison.png")
plt.close()
print("Saved: llm_radar_comparison.png")

# ============================================================
# 7. Combined heatmap (models x metrics)
# ============================================================
import matplotlib.colors as mcolors
heat_metrics = ["accuracy", "precision", "recall", "f1", "groundedness"]
heat_labels = ["Accuracy", "Precision", "Recall", "F1", "Groundedness"]
heat_data = np.array([[results[m][k] for k in heat_metrics] for m in models])

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(heat_data, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(heat_labels)))
ax.set_xticklabels(heat_labels)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)
for i in range(len(models)):
    for j in range(len(heat_labels)):
        ax.text(j, i, f"{heat_data[i,j]:.1f}", ha="center", va="center",
                color="black" if heat_data[i,j] > 50 else "white", fontweight="bold")
ax.set_title("Overall LLM+RAG Performance Heatmap", fontweight="bold")
plt.colorbar(im, label="Score (%)")
plt.tight_layout()
plt.savefig("outputs/combined_analysis/performance_heatmap.png")
plt.close()
print("Saved: performance_heatmap.png")

# ============================================================
# 8. System architecture diagram (RAG pipeline, no DL classifier boxes)
# ============================================================
import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(15, 4))
stages = [
    ("User\nQuery", "#93c5fd"),
    ("Hybrid Retrieval\n(BM25 + Dense\nEmbeddings)", "#2563eb"),
    ("Reciprocal Rank\nFusion (RRF)", "#1d4ed8"),
    ("Cross-Encoder\nReranking", "#0891b2"),
    ("LLM Generation\n(Ollama)", "#10b981"),
    ("Groundedness\nVerification (NLI)", "#f59e0b"),
    ("Final\nAnswer", "#93c5fd"),
]
x_positions = [i * 2.1 for i in range(len(stages))]
y = 0.5
for i, (label, color) in enumerate(stages):
    box = mpatches.FancyBboxPatch((x_positions[i], y - 0.28), 1.8, 0.56, boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(box)
    ax.text(x_positions[i] + 0.9, y, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    if i < len(stages) - 1:
        ax.annotate("", xy=(x_positions[i+1], y), xytext=(x_positions[i] + 1.8, y),
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8))
ax.set_xlim(-0.3, x_positions[-1] + 2.1)
ax.set_ylim(0, 1)
ax.set_title("Hybrid RAG Pipeline Architecture (LLM-Focused View)", fontsize=14, fontweight="bold", pad=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/architecture/rag_pipeline_architecture.png", bbox_inches="tight")
plt.close()
print("Saved: rag_pipeline_architecture.png")

print("\n" + "="*60)
print("IMPORTANT LIMITATIONS (per your no-fabrication requirement):")
print("="*60)
print("1. Perplexity was NOT computed. Ollama's chat API does not expose")
print("   token-level log-probabilities required for perplexity calculation.")
print("   This is a genuine technical limitation, not an omission.")
print("2. Training/validation loss curves are NOT applicable to the LLMs")
print("   (llama3.2, gemma2, gemma3) since they are used pretrained via")
print("   Ollama and were not fine-tuned/trained in this project.")
print("3. RAG-vs-No-RAG comparison charts require a separate evaluation")
print("   run (not yet performed) — let me know if you want this next.")
print("\nAll LLM/RAG figures saved under outputs/ in the requested folder structure.")
