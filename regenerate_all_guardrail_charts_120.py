import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from rag_engine import LegalRAGEngine

with open("large_guardrail_test_set.json") as f:
    test_set = json.load(f)

engine = LegalRAGEngine(model_name="llama3.2:1b")

y_true = []
distances = []

print(f"Collecting distances for {len(test_set)} questions...")
for i, item in enumerate(test_set):
    _, is_confident, min_distance, predicted_class = engine.retrieve(item["question"])
    y_true.append(item["expected"])
    distances.append(min_distance)
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(test_set)} done")

y_true = np.array(y_true)
distances = np.array(distances)

# For ROC/PR: lower distance = more likely "in_domain" (label=1), so use NEGATIVE distance as the "score"
scores = -distances

# --- ROC Curve ---
fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Retrieval Guardrail - ROC Curve (n={len(test_set)})", fontsize=13, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curve_guardrail_120.png", dpi=150)
plt.close()
print(f"\nROC AUC: {roc_auc:.3f}")
print("Saved: outputs/roc_curve_guardrail_120.png")

# --- Precision-Recall Curve ---
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, scores)
avg_precision = average_precision_score(y_true, scores)

plt.figure(figsize=(7, 6))
plt.plot(recall_vals, precision_vals, color="#10b981", lw=2, label=f"PR curve (AP = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Retrieval Guardrail - Precision-Recall Curve (n={len(test_set)})", fontsize=13, fontweight="bold")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("outputs/pr_curve_guardrail_120.png", dpi=150)
plt.close()
print(f"Average Precision: {avg_precision:.3f}")
print("Saved: outputs/pr_curve_guardrail_120.png")

# --- Threshold Sensitivity ---
thresh_range = np.arange(distances.min() - 0.05, distances.max() + 0.05, 0.01)
accs, precs, recs, f1s = [], [], [], []

for t in thresh_range:
    preds = (distances <= t).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
    accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

plt.figure(figsize=(10, 6))
plt.plot(thresh_range, accs, label="Accuracy", color="#3b82f6")
plt.plot(thresh_range, precs, label="Precision", color="#f59e0b")
plt.plot(thresh_range, recs, label="Recall", color="#10b981")
plt.plot(thresh_range, f1s, label="F1 Score", color="#ef4444")
plt.axvline(x=1.35, linestyle="--", color="gray", label="Chosen threshold (1.35)")
plt.xlabel("Distance Threshold")
plt.ylabel("Score")
plt.title(f"Guardrail Threshold Sensitivity (n={len(test_set)})", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/threshold_sensitivity_guardrail_120.png", dpi=150)
plt.close()
print("Saved: outputs/threshold_sensitivity_guardrail_120.png")

# --- Diagnostic Metrics (Sensitivity/Specificity/PPV/NPV/LR+/LR-) at chosen threshold ---
preds_at_threshold = (distances <= 1.35).astype(int)
TP = np.sum((preds_at_threshold == 1) & (y_true == 1))
FP = np.sum((preds_at_threshold == 1) & (y_true == 0))
FN = np.sum((preds_at_threshold == 0) & (y_true == 1))
TN = np.sum((preds_at_threshold == 0) & (y_true == 0))

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

print(f"\nConfusion @ threshold=1.35: TP={TP} FP={FP} FN={FN} TN={TN}")
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
plt.title(f"Retrieval Guardrail - Diagnostic Metrics (n={len(test_set)}, threshold=1.35)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/diagnostic_metrics_guardrail_120.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/diagnostic_metrics_guardrail_120.png")

print("\nAll guardrail charts regenerated from the same 120-question dataset.")
