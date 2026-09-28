"""
System architecture diagram (paper "Figure 1" style) for the Legal AI RAG
pipeline. Pure matplotlib, no external dependencies like graphviz needed.
Edit the STAGES list below to exactly match your real pipeline.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

STAGES = [
    "User Query",
    "Query Domain\nClassifier",
    "Hybrid Retrieval\n(BM25 + Dense Embeddings)",
    "Cross-Encoder\nReranker",
    "Retrieval\nGuardrail",
    "LLM Generation\n(Ollama / llama3.2)",
    "Final Answer",
]

fig, ax = plt.subplots(figsize=(12, 3.2))
ax.set_xlim(0, len(STAGES))
ax.set_ylim(0, 1)
ax.axis("off")

box_w, box_h = 0.85, 0.5
colors = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#f59e0b", "#10b981", "#93c5fd"]

for i, (stage, color) in enumerate(zip(STAGES, colors)):
    x = i + (1 - box_w) / 2
    y = 0.25
    box = FancyBboxPatch((x, y), box_w, box_h,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           linewidth=1.5, edgecolor="#1e293b", facecolor=color)
    ax.add_patch(box)
    ax.text(i + 0.5, y + box_h / 2, stage, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white" if i not in (0, 6) else "#1e293b")

    if i < len(STAGES) - 1:
        arrow = FancyArrowPatch((i + box_w + (1 - box_w) / 2, y + box_h / 2),
                                  (i + 1 + (1 - box_w) / 2, y + box_h / 2),
                                  arrowstyle="-|>", mutation_scale=15, color="#1e293b", linewidth=1.5)
        ax.add_patch(arrow)

plt.title("Legal AI RAG - System Architecture", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/system_architecture_diagram.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: outputs/system_architecture_diagram.png")
