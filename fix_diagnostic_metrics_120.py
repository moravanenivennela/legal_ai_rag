import json
import numpy as np
import matplotlib.pyplot as plt
from rag_engine import LegalRAGEngine

with open("large_guardrail_test_set.json") as f:
    test_set = json.load(f)

engine = LegalRAGEngine(model_name="llama3.2:1b")

y_true, y_pred = [], []
for item in test_set:
    _, is_confident, _, _ = engine.retrieve(item["question"])  # uses BOTH guardrail layers correctly
    y_true.append(item["expected"])
    y_pred.append(1 if is_confident else 0)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

TP = np.sum((y_pred == 1) & (y_true == 1))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))

def compute_diag(TP, FN, FP, TN, name):
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    lr_plus = sens / (1 - spec) if (1 - spec) > 0 else float('inf')
    lr_minus = (1 - sens) / spec if spec > 0 else float('inf')
    return [name, f"{sens*100:.1f}%", f"{spec*100:.1f}%", f"{ppv*100:.1f}%", f"{npv*100:.1f}%",
            f"{lr_plus:.2f}" if lr_plus != float('inf') else "inf",
            f"{lr_minus:.2f}" if lr_minus != float('inf') else "inf"]

row_in = compute_diag(TP, FN, FP, TN, "in_domain")
row_out = compute_diag(TN, FP, FN, TP, "out_of_domain")

print(f"Confusion (correct, both guardrail layers): TP={TP} FP={FP} FN={FN} TN={TN}")
print(row_in)
print(row_out)

fig, ax = plt.subplots(figsize=(10, 2.2))
ax.axis("off")
col_labels = ["Class", "Sensitivity", "Specificity", "PPV", "NPV", "LR+", "LR-"]
rows = [row_out, row_in]
table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor("#3b82f6")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
plt.title(f"Retrieval Guardrail - Diagnostic Metrics (n={len(test_set)}, full guardrail)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/diagnostic_metrics_guardrail_120.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved corrected: outputs/diagnostic_metrics_guardrail_120.png")
