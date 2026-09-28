"""
Ablation study bar chart: compares system variants (e.g. BM25-only,
Dense-only, Hybrid, Hybrid+Guardrail) on Accuracy/F1. Fill in
ABLATION_RESULTS with numbers you actually measure by running your
evaluation script once per configuration.
"""
import matplotlib.pyplot as plt
import numpy as np

ABLATION_RESULTS = {
    "BM25 only":             {"Accuracy": 78.4, "F1": 74.1},
    "Dense only":            {"Accuracy": 84.6, "F1": 81.2},
    "Hybrid (no guardrail)": {"Accuracy": 88.9, "F1": 86.0},
    "Hybrid + Guardrail":    {"Accuracy": 90.7, "F1": 91.8},
}

labels = list(ABLATION_RESULTS.keys())
acc_vals = [ABLATION_RESULTS[k]["Accuracy"] for k in labels]
f1_vals = [ABLATION_RESULTS[k]["F1"] for k in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy", color="#3b82f6")
bars2 = ax.bar(x + width / 2, f1_vals, width, label="F1 Score", color="#10b981")

for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=10, ha="right")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.set_title("Ablation Study - Effect of Retrieval Strategy & Guardrail", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/ablation_comparison.png", dpi=150)
plt.close()

print("Saved: outputs/ablation_comparison.png")
