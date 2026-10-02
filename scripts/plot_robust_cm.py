import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm = np.load("outputs/confusion_matrix_500.npy")
labels = ["Constitution", "Consumer Protection", "Out-of-Domain"]

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=labels, yticklabels=labels, cbar=True, annot_kws={"size": 13})

plt.title(f"Fig 3. Domain Guardrail Confusion Matrix (N = {cm.sum()} Evaluation Queries)", fontsize=11, fontweight='bold')
plt.xlabel("Predicted Class", fontsize=10, fontweight='bold')
plt.ylabel("Actual Class", fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/confusion_matrix_500.png", dpi=300)

print("Plot saved to outputs/plots/confusion_matrix_500.png")
