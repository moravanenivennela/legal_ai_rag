import os, sys, glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from rag_engine import LegalRAGEngine

os.makedirs("outputs", exist_ok=True)

search_dirs = [".", "..", "evaluation_baseline", "../evaluation_baseline", "backups", "../backups"]
found_files = []
for d in search_dirs:
    found_files.extend(glob.glob(os.path.join(d, "*.csv")))
    found_files.extend(glob.glob(os.path.join(d, "*/*.csv")))

csv_path = None
for path in set(found_files):
    filename = os.path.basename(path).lower()
    if "500" in filename or "evaluation" in filename or "domain" in filename or "blind" in filename:
        csv_path = path
        break

if not csv_path and found_files:
    csv_path = found_files[0]

if not csv_path:
    print("ERROR: No CSV files found in project directories.")
    sys.exit(1)

print("Loading dataset from:", csv_path)
df = pd.read_csv(csv_path)

query_col = next((c for c in ["query", "question", "prompt", "user_query", "text"] if c in df.columns), None)
target_col = next((c for c in ["expected_source", "ground_truth", "domain", "target", "label"] if c in df.columns), None)

if not query_col or not target_col:
    print("ERROR: CSV columns do not match expected format.", list(df.columns))
    sys.exit(1)

mapping = {"constitution_of_india.pdf": "constitution", "consumer_protection_act_2019.pdf": "consumer_protection"}
y_true = df[target_col].astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, x))

engine = LegalRAGEngine()
print("Evaluating", len(df), "queries through V3 engine...")
v3_preds = [engine.query_classifier.predict([engine.embedding_fn.embed_query(str(q))])[0] for q in df[query_col]]

df["predicted_domain_v3"] = v3_preds
df.to_csv("outputs/RAG_RETRIEVAL_EVALUATION_500_v3.csv", index=False)

acc = accuracy_score(y_true, v3_preds)
p, r, f1, _ = precision_recall_fscore_support(y_true, v3_preds, average="weighted", zero_division=0)
rep = classification_report(y_true, v3_preds, zero_division=0)

summary = "=" * 42 + chr(10) + "   FRESH EVALUATION METRICS (V3 ENGINE)" + chr(10) + "=" * 42 + chr(10)
summary += f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)" + chr(10)
summary += f"Precision (PPV):     {p:.4f}" + chr(10)
summary += f"Recall (Sensitivity): {r:.4f}" + chr(10)
summary += f"F1-Score:            {f1:.4f}" + chr(10)
summary += "=" * 42 + chr(10) + chr(10) + "--- Classification Report ---" + chr(10) + rep

with open("outputs/evaluation_metrics_v3.txt", "w", encoding="utf-8") as mf:
    mf.write(summary)

print(chr(10) + summary)

labels = sorted(list(set(y_true).union(set(v3_preds))))
cm = confusion_matrix(y_true, v3_preds, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Fresh Domain Router Confusion Matrix (V3)")
plt.xlabel("Predicted Domain")
plt.ylabel("Actual Domain")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_v3.png", dpi=300)
plt.close()
print("Saved fresh plot to outputs/confusion_matrix_v3.png")