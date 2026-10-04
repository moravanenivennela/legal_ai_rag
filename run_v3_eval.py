import pandas as pd
import pickle
from rag_engine import LegalRAGEngine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

engine = LegalRAGEngine()
csv_path = "outputs/RAG_RETRIEVAL_EVALUATION_500.csv"
df = pd.read_csv(csv_path)

mapping = {
    "constitution_of_india.pdf": "constitution",
    "consumer_protection_act_2019.pdf": "consumer_protection"
}
y_true = df["expected_source"].astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, x))

print("Evaluating 500 queries through V3 engine...")
v3_predictions = []
for q in df["query"]:
    vec = engine.embedding_fn.embed_query(q)
    pred = engine.query_classifier.predict([vec])[0]
    v3_predictions.append(pred)

df["predicted_domain_v3"] = v3_predictions
output_csv = "outputs/RAG_RETRIEVAL_EVALUATION_500_v3.csv"
df.to_csv(output_csv, index=False)
print(f"Updated predictions saved to {output_csv}\n")

acc = accuracy_score(y_true, v3_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, v3_predictions, average="weighted", zero_division=0)

print("==========================================")
print("   UPDATED EVALUATION METRICS (V3 CLASSIFIER) ")
print("==========================================")
print(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (PPV):     {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score:            {f1:.4f}")
print("==========================================\n")
print("--- Detailed Classification Report ---")
print(classification_report(y_true, v3_predictions, zero_division=0))
