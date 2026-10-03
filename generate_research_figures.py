from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path("outputs 2")
OUT.mkdir(parents=True, exist_ok=True)

# Publication-friendly defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def save_table_image(filename, title, columns, rows, col_widths=None,
                     font_size=9, scale_y=1.5):
    """Render a data table as a high-resolution PNG."""
    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.42 * (len(rows) + 2))))
    ax.axis("off")
    ax.set_title(title, pad=16, fontweight="bold")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")

    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------
# FIGURE 1: Row-normalized domain-classification confusion matrix
# Recorded raw matrix:
#                 Predicted: Constitution, Consumer, OOD
# True Constitution                         30,  0,  0
# True Consumer Protection                   0, 30,  0
# True Out-of-domain                         1,  0, 39
# Normalize each true-class row to percentages so the figure
# displays rates, not raw evaluation counts.
# ---------------------------------------------------------
raw_matrix = np.array([
    [30, 0, 0],
    [0, 30, 0],
    [1, 0, 39],
], dtype=float)

row_totals = raw_matrix.sum(axis=1, keepdims=True)
normalized = np.divide(
    raw_matrix,
    row_totals,
    out=np.zeros_like(raw_matrix),
    where=row_totals != 0,
) * 100.0

class_labels = [
    "Constitution",
    "Consumer Protection",
    "Out-of-domain",
]

fig, ax = plt.subplots(figsize=(7.2, 5.8))
im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
ax.set_title("Domain Classification Confusion Matrix", pad=14, fontweight="bold")
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_xticks(range(len(class_labels)), labels=class_labels, rotation=18, ha="right")
ax.set_yticks(range(len(class_labels)), labels=class_labels)

for i in range(normalized.shape[0]):
    for j in range(normalized.shape[1]):
        value = normalized[i, j]
        ax.text(
            j, i, f"{value:.1f}%",
            ha="center", va="center",
            color="white" if value > 55 else "black",
            fontweight="bold",
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized percentage")
fig.tight_layout()
fig.savefig(
    OUT / "01_domain_confusion_matrix.png",
    bbox_inches="tight", facecolor="white"
)
plt.close(fig)


# ---------------------------------------------------------
# FIGURE 2: Classification metrics
# Values are the recorded final frozen blind classification
# results. No support/count column is included.
# ---------------------------------------------------------
classification_rows = [
    ["Constitution", "96.77%", "100.00%", "98.36%"],
    ["Consumer Protection", "100.00%", "100.00%", "100.00%"],
    ["Out-of-domain", "100.00%", "97.50%", "98.73%"],
    ["Macro average", "98.92%", "99.17%", "99.03%"],
    ["Weighted average", "99.03%", "99.00%", "99.00%"],
    ["Overall accuracy", "—", "—", "99.00%"],
]

save_table_image(
    "02_classification_metrics.png",
    "Domain Classification Performance",
    ["Class / Average", "Precision", "Recall", "F1-score"],
    classification_rows,
    col_widths=[0.40, 0.20, 0.20, 0.20],
    font_size=9,
    scale_y=1.55,
)


# ---------------------------------------------------------
# FIGURE 3: Retrieval ablation
# Recorded Hit@K percentages for each retrieval configuration.
# ---------------------------------------------------------
retrieval_rows = [
    ["Dense retrieval", "100.00%", "100.00%", "100.00%"],
    ["BM25 retrieval", "90.00%", "96.67%", "96.67%"],
    ["Hybrid RRF", "100.00%", "100.00%", "100.00%"],
    ["Full reranker", "100.00%", "100.00%", "100.00%"],
]

save_table_image(
    "03_retrieval_ablation.png",
    "Retrieval Ablation Results",
    ["Retrieval configuration", "Hit@1", "Hit@3", "Hit@4"],
    retrieval_rows,
    col_widths=[0.40, 0.20, 0.20, 0.20],
    font_size=9,
    scale_y=1.8,
)


# ---------------------------------------------------------
# FIGURE 4: Answer-generation evaluation
# Recorded results. Source match is NOT legal correctness.
# ---------------------------------------------------------
answer_rows = [
    ["Successful answer generation", "Completed for all evaluated queries"],
    ["Expected-source match", "100.00%"],
    ["Average NLI-based groundedness", "45.02%"],
    ["Answers with groundedness >= 70%", "26.67%"],
    ["Average response latency", "14.73 seconds"],
]

save_table_image(
    "04_answer_generation_evaluation.png",
    "Answer-Generation Evaluation Results",
    ["Evaluation measure", "Observed result"],
    answer_rows,
    col_widths=[0.48, 0.52],
    font_size=9,
    scale_y=1.8,
)

print("\nFigure generation complete.")
print(f"Output folder: {OUT.resolve()}")
for path in sorted(OUT.glob("*.png")):
    print(f"{path.name}  |  {path.stat().st_size / 1024:.1f} KB")
