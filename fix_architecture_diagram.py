import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(15, 4.5))

stages = [
    ("User\nQuery", "#93c5fd"),
    ("Query Domain\nClassifier\n(Guardrail 1)", "#3b82f6"),
    ("Hybrid Retrieval\n(BM25 + Dense\nEmbeddings)", "#2563eb"),
    ("Distance\nThreshold\n(Guardrail 2)", "#f59e0b"),
    ("Cross-Encoder\nReranker", "#1d4ed8"),
    ("LLM Generation\n(Ollama /\nllama3.2)", "#10b981"),
    ("Final\nAnswer", "#93c5fd"),
]

x_positions = [i * 2.1 for i in range(len(stages))]
y = 0.5

for i, (label, color) in enumerate(stages):
    box = mpatches.FancyBboxPatch((x_positions[i], y - 0.28), 1.8, 0.56,
                                    boxstyle="round,pad=0.03", facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(box)
    ax.text(x_positions[i] + 0.9, y, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    if i < len(stages) - 1:
        ax.annotate("", xy=(x_positions[i+1], y), xytext=(x_positions[i] + 1.8, y),
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8))

ax.set_xlim(-0.3, x_positions[-1] + 2.1)
ax.set_ylim(0, 1)
ax.set_title("Legal AI RAG - System Architecture", fontsize=15, fontweight="bold", pad=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/system_architecture_diagram_fixed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/system_architecture_diagram_fixed.png")
