import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from rag_engine import LegalRAGEngine

with open("large_guardrail_test_set.json") as f:
    test_set = json.load(f)

engine = LegalRAGEngine(model_name="llama3.2:1b")

y_true, y_pred = [], []
print(f"Running guardrail evaluation on {len(test_set)} questions...\n")

for i, item in enumerate(test_set):
    _, is_confident, min_distance, predicted_class = engine.retrieve(item["question"])
    pred_label = 1 if is_confident else 0
    y_true.append(item["expected"])
    y_pred.append(pred_label)
    status = "OK" if pred_label == item["expected"] else "MISS"
    print(f"[{i+1:3d}/{len(test_set)}] {status:4s} true={item['expected']} pred={pred_label} dist={min_distance:.3f} | {item['question'][:55]}")

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\n{'='*70}")
print(f"RETRIEVAL GUARDRAIL — Evaluation on {len(test_set)} questions")
print(f"{'='*70}")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-score:  {f1*100:.2f}%")
print(classification_report(y_true, y_pred, target_names=["out_of_domain", "in_domain"], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["out_of_domain", "in_domain"], yticklabels=["out_of_domain", "in_domain"])
plt.title(f"Retrieval Guardrail - Confusion Matrix (n={len(test_set)})", fontsize=13, fontweight="bold")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_guardrail_120.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
names = ["Accuracy", "Precision", "Recall", "F1 Score"]
vals = [acc*100, prec*100, rec*100, f1*100]
bars = plt.bar(names, vals, color="#c5a059")
for bar, val in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.2f}%", ha="center", fontweight="bold")
plt.ylim(0, 105)
plt.title(f"Retrieval Guardrail Performance (n={len(test_set)} questions)", fontsize=13, fontweight="bold")
plt.ylabel("Score (%)")
plt.tight_layout()
plt.savefig("outputs/evaluation_metrics_guardrail_120.png", dpi=150)
plt.close()
print("\nSaved: outputs/confusion_matrix_guardrail_120.png")
print("Saved: outputs/evaluation_metrics_guardrail_120.png")
