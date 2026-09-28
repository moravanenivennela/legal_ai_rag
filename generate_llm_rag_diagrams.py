import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'

# ============================================================
# Diagram 1: Flowchart — How each LLM is evaluated through RAG
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))

stages = [
    ("Legal\nQuestion", "#9CA3AF"),
    ("Hybrid Retrieval\n(BM25 + Dense + RRF)", "#3b82f6"),
    ("Cross-Encoder\nRerank (top-6)", "#3b82f6"),
    ("LLM Generation\n(swap: 7 models tested)", "#c5a059"),
    ("Semantic F1 vs.\nReference Answer", "#10b981"),
    ("Accuracy / Precision /\nRecall / F1 per model", "#9333EA"),
]

x_positions = [i * 2.0 for i in range(len(stages))]
y = 0.5

for i, (label, color) in enumerate(stages):
    box = mpatches.FancyBboxPatch((x_positions[i], y - 0.22), 1.7, 0.44,
                                    boxstyle="round,pad=0.03", facecolor=color, edgecolor="black", linewidth=1)
    ax.add_patch(box)
    ax.text(x_positions[i] + 0.85, y, label, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    if i < len(stages) - 1:
        ax.annotate("", xy=(x_positions[i+1], y), xytext=(x_positions[i] + 1.7, y),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.8))

ax.set_xlim(-0.3, x_positions[-1] + 2.0)
ax.set_ylim(0, 1)
ax.set_title("Data Flow: Evaluating Multiple LLMs Through the Same RAG Pipeline", fontsize=13, fontweight="bold", pad=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/llm_rag_evaluation_flowchart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/llm_rag_evaluation_flowchart.png")


# ============================================================
# Diagram 2: Radar chart comparing all 7 models across 4 metrics
# ============================================================
models_data = {
    "llama3.2:1b":      [100.0, 69.4, 83.5, 75.6],
    "llama3.2:3b":      [90.0, 72.3, 83.0, 77.1],
    "phi3:mini":        [80.0, 64.4, 82.2, 71.7],
    "phi4-mini":        [90.0, 68.3, 83.1, 74.6],
    "gemma2:2b":        [100.0, 70.0, 83.5, 76.0],
    "gemma3:4b":        [100.0, 73.1, 81.9, 76.9],
    "qwen2.5:3b":       [80.0, 65.1, 77.9, 70.8],
}
categories = ["Accuracy", "Precision", "Recall", "F1-score"]
n_cats = len(categories)

angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
colors = ["#3b82f6", "#c5a059", "#ef4444", "#06b6d4", "#10b981", "#9333EA", "#f97316"]

for i, (model, values) in enumerate(models_data.items()):
    vals = values + values[:1]
    ax.plot(angles, vals, linewidth=2, label=model, color=colors[i % len(colors)])
    ax.fill(angles, vals, alpha=0.05, color=colors[i % len(colors)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.set_title("RAG Performance Comparison Across LLMs (Radar View)", fontsize=13, fontweight="bold", pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig("outputs/llm_rag_radar_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/llm_rag_radar_comparison.png")


# ============================================================
# Diagram 3: Speed vs Quality tradeoff scatter
# ============================================================
speed_data = {
    "llama3.2:1b": (36.39, 75.6),
    "llama3.2:3b": (54.40, 77.1),
    "phi3:mini": (461.45, 71.7),
    "phi4-mini": (68.39, 74.6),
    "gemma2:2b": (38.48, 76.0),
    "gemma3:4b": (60.39, 76.9),
    "qwen2.5:3b": (48.86, 70.8),
}

fig, ax = plt.subplots(figsize=(10, 7))
for i, (model, (time_val, f1_val)) in enumerate(speed_data.items()):
    ax.scatter(time_val, f1_val, s=200, color=colors[i % len(colors)], edgecolor="black", linewidth=1, zorder=3)
    ax.annotate(model, (time_val, f1_val), xytext=(8, 8), textcoords="offset points", fontsize=10, fontweight="bold")

ax.set_xlabel("Average Response Time (seconds, log scale)", fontsize=12)
ax.set_ylabel("F1-score (%)", fontsize=12)
ax.set_xscale("log")
ax.set_title("Speed vs. Quality Tradeoff Across LLM Models\n(Lower-left = fast & accurate)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/llm_speed_vs_quality.png", dpi=150)
plt.close()
print("Saved: outputs/llm_speed_vs_quality.png")

print("\nAll 3 diagrams generated successfully.")
