import json, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from langchain_huggingface import HuggingFaceEmbeddings

plt.rcParams['savefig.dpi'] = 300

with open("expanded_classifier_data.json") as f:
    ALL_DATA = json.load(f)

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
texts = [t[0] for t in ALL_DATA]
labels = [t[1] for t in ALL_DATA]
X_full = np.array(embedder.embed_documents(texts))
y_full = np.array(labels)

# ============================================================
# 1. Per-class F1 breakdown (not just macro-averaged)
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000)
y_pred = cross_val_predict(clf, X_full, y_full, cv=cv)

class_labels = sorted(set(y_full))
per_class_f1 = f1_score(y_full, y_pred, labels=class_labels, average=None)
per_class_prec = precision_score(y_full, y_pred, labels=class_labels, average=None)
per_class_rec = recall_score(y_full, y_pred, labels=class_labels, average=None)

x = np.arange(len(class_labels))
width = 0.25
fig, ax = plt.subplots(figsize=(9,6))
ax.bar(x - width, per_class_prec*100, width, label="Precision", color="#3b82f6")
ax.bar(x, per_class_rec*100, width, label="Recall", color="#c5a059")
ax.bar(x + width, per_class_f1*100, width, label="F1-score", color="#10b981")
ax.set_xticks(x); ax.set_xticklabels(class_labels)
ax.set_ylabel("Score (%)"); ax.set_ylim(0,105)
ax.set_title(f"Query Classifier - Per-Class Performance (n={len(ALL_DATA)})", fontweight="bold")
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("dl_outputs/per_class_f1_query_classifier.png")
plt.close()
print("Saved: per_class_f1_query_classifier.png")

# ============================================================
# 2. Learning curve: accuracy vs training set size
# ============================================================
sizes = [45, 75, 105, len(ALL_DATA)]
sizes = sorted(set([s for s in sizes if s <= len(ALL_DATA)]))
learning_curve_acc = []

for size in sizes:
    idx = np.random.RandomState(42).choice(len(X_full), size=size, replace=False)
    X_sub, y_sub = X_full[idx], y_full[idx]
    try:
        cv_sub = StratifiedKFold(n_splits=min(5, min(np.unique(y_sub, return_counts=True)[1])), shuffle=True, random_state=42)
        y_pred_sub = cross_val_predict(LogisticRegression(max_iter=1000), X_sub, y_sub, cv=cv_sub)
        acc_sub = (y_pred_sub == y_sub).mean() * 100
    except Exception as e:
        acc_sub = None
    learning_curve_acc.append(acc_sub)
    print(f"  size={size} -> accuracy={acc_sub}")

valid_points = [(s,a) for s,a in zip(sizes, learning_curve_acc) if a is not None]
plt.figure(figsize=(8,5))
xs = [p[0] for p in valid_points]; ys = [p[1] for p in valid_points]
plt.plot(xs, ys, marker='o', linewidth=2, markersize=8, color="#3b82f6")
for x_, y_ in zip(xs, ys):
    plt.annotate(f"{y_:.1f}%", (x_, y_), textcoords="offset points", xytext=(0,8), ha='center', fontweight='bold')
plt.xlabel("Training Set Size (number of examples)")
plt.ylabel("Cross-Validated Accuracy (%)")
plt.title("Query Classifier Learning Curve: Dataset Size vs. Accuracy", fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dl_outputs/learning_curve_query_classifier.png")
plt.close()
print("Saved: learning_curve_query_classifier.png")

# ============================================================
# 3. Combined summary table (pulls from available result files + known constants)
# ============================================================
summary_rows = []

# Query classifier (just computed above, full dataset)
acc = (y_pred == y_full).mean() * 100
prec = precision_score(y_full, y_pred, average='macro') * 100
rec = recall_score(y_full, y_pred, average='macro') * 100
f1 = f1_score(y_full, y_pred, average='macro') * 100
summary_rows.append(["Query Domain Classifier", f"{acc:.1f}%", f"{prec:.1f}%", f"{rec:.1f}%", f"{f1:.1f}%"])

# CNN (hardcoded from last confirmed real run - update if retrained)
summary_rows.append(["CNN Image Classifier", "96.9%", "96.8%", "96.8%", "96.8%"])

# Retrieval Guardrail (hardcoded from confirmed 120-question run)
summary_rows.append(["Retrieval Guardrail (n=120)", "93.3%", "100.0%", "90.0%", "94.7%"])

# LLM comparison, if results file exists
if os.path.exists("llm_results_incremental.json"):
    with open("llm_results_incremental.json") as f:
        llm_data = json.load(f)
    for model_name, m in llm_data.items():
        summary_rows.append([f"LLM: {model_name} (RAG)", f"{m['accuracy']:.1f}%", f"{m['precision']:.1f}%", f"{m['recall']:.1f}%", f"{m['f1']:.1f}%"])

fig, ax = plt.subplots(figsize=(11, 0.6*len(summary_rows)+1.5))
ax.axis("off")
col_labels = ["Component", "Accuracy", "Precision", "Recall", "F1-score"]
table = ax.table(cellText=summary_rows, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
for j in range(len(col_labels)):
    table[(0,j)].set_facecolor("#1d4ed8")
    table[(0,j)].set_text_props(color="white", fontweight="bold")
plt.title("Combined System Performance Summary", fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/combined_analysis/full_system_summary_table.png", bbox_inches="tight")
plt.close()
print("Saved: outputs/combined_analysis/full_system_summary_table.png")
