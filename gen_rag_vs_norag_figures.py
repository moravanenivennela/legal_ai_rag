import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

with open("rag_vs_norag_incremental.json") as f:
    data = json.load(f)

models = list(data.keys())
print("Models:", models)

# ============================================================
# 1. Grouped bar chart: RAG vs No-RAG across all 4 metrics
# ============================================================
metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

for model_name in models:
    rag_vals = [data[model_name]["rag"][m] for m in metrics_to_plot]
    norag_vals = [data[model_name]["norag"][m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, rag_vals, width, label="With RAG", color="#10b981")
    bars2 = ax.bar(x + width/2, norag_vals, width, label="Without RAG (Baseline)", color="#ef4444")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 105)
    ax.set_title(f"RAG vs. No-RAG Performance — {model_name} (n=30)", fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    safe_name = model_name.replace(":", "_")
    plt.savefig(f"outputs/rag_performance/rag_vs_norag_{safe_name}.png")
    plt.close()
    print(f"Saved: rag_vs_norag_{safe_name}.png")

# ============================================================
# 2. Combined comparison across both models
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metrics_to_plot))
width = 0.2
colors_rag = ["#10b981", "#059669"]
colors_norag = ["#f87171", "#ef4444"]

for i, model_name in enumerate(models):
    rag_vals = [data[model_name]["rag"][m] for m in metrics_to_plot]
    norag_vals = [data[model_name]["norag"][m] for m in metrics_to_plot]
    ax.bar(x + (i*2-1.5)*width, rag_vals, width, label=f"{model_name} (RAG)", color=colors_rag[i % len(colors_rag)])
    ax.bar(x + (i*2-0.5)*width, norag_vals, width, label=f"{model_name} (No-RAG)", color=colors_norag[i % len(colors_norag)])

ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 105)
ax.set_title("RAG vs. No-RAG Performance Across Models (n=30)", fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/rag_performance/rag_vs_norag_combined.png")
plt.close()
print("Saved: rag_vs_norag_combined.png")

# ============================================================
# 3. Confusion Matrix + Specificity/PPV
# Framing: RAG answers = expected "correct" (positive class),
#          No-RAG answers = expected "less reliable" (negative class).
# A prediction is "correct" if per-question accuracy flag == 1.
# ============================================================
def compute_diag(TP, FN, FP, TN):
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    return sens, spec, ppv, npv

for model_name in models:
    rag_accs = data[model_name]["raw_rag_accs"]
    norag_accs = data[model_name]["raw_norag_accs"]

    # Positive class = RAG (expected correct=1), Negative class = No-RAG (expected correct=0)
    y_true = [1]*len(rag_accs) + [0]*len(norag_accs)
    y_pred = [int(a) for a in rag_accs] + [int(a) for a in norag_accs]

    TP = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    FN = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    FP = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    TN = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)

    sens, spec, ppv, npv = compute_diag(TP, FN, FP, TN)

    # Confusion matrix plot
    cm = np.array([[TN, FP],[FN, TP]])
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["No-RAG\n(expected less reliable)", "RAG\n(expected correct)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(["Predicted\nIncorrect", "Predicted\nCorrect"])
    ax.set_yticks([0,1]); ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"Legal Answer Correctness — Confusion Matrix\n{model_name} (RAG vs. No-RAG framing)", fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(":", "_")
    plt.savefig(f"outputs/evaluation_metrics/confusion_matrix_correctness_{safe_name}.png")
    plt.close()

    # Specificity/PPV table
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.axis("off")
    col_labels = ["Sensitivity", "Specificity", "PPV", "NPV"]
    row = [f"{sens*100:.1f}%", f"{spec*100:.1f}%", f"{ppv*100:.1f}%", f"{npv*100:.1f}%"]
    table = ax.table(cellText=[row], colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 2.2)
    for j in range(len(col_labels)):
        table[(0,j)].set_facecolor("#3b82f6")
        table[(0,j)].set_text_props(color="white", fontweight="bold")
    plt.title(f"Legal Answer Correctness — Diagnostic Metrics\n{model_name}", fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"outputs/evaluation_metrics/diagnostic_metrics_correctness_{safe_name}.png", bbox_inches="tight")
    plt.close()

    print(f"\n{model_name}: TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"  Sensitivity={sens*100:.1f}% Specificity={spec*100:.1f}% PPV={ppv*100:.1f}% NPV={npv*100:.1f}%")

print("\nAll RAG-vs-No-RAG figures saved.")
