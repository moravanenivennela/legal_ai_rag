import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from rag_engine import LegalRAGEngine

# 1 = should answer (in-domain: Constitution or Consumer Protection Act)
# 0 = should reject (out-of-domain)
GUARDRAIL_TEST_SET = [
    # --- Clearly in-domain: Constitution ---
    ("What are the six consumer rights under the Consumer Protection Act?", 1),
    ("What is the pecuniary jurisdiction of the District Commission?", 1),
    ("What is Product Liability under the 2019 Act?", 1),
    ("What does Article 21 of the Constitution protect?", 1),
    ("Explain the writ jurisdiction under Article 32.", 1),
    ("How does Article 246 distribute legislative powers?", 1),
    ("What are the Directive Principles of State Policy?", 1),
    ("What is the significance of the Preamble?", 1),
    ("What is Article 14 about equality before law?", 1),
    ("Explain the concept of judicial review under the Constitution.", 1),
    ("What fundamental duties does a citizen have?", 1),
    ("How is the Union List different from the State List?", 1),
    ("What powers does the President have under the Constitution?", 1),
    ("What is Article 19 and what freedoms does it protect?", 1),
    ("How is the Supreme Court's jurisdiction defined?", 1),
    ("What is the procedure to amend the Constitution?", 1),
    ("What is the difference between Article 32 and Article 226?", 1),
    ("What are Fundamental Rights under Part III?", 1),
    ("Explain the federal structure created by the Constitution.", 1),
    ("What is the role of the Governor under the Constitution?", 1),
    # --- Clearly in-domain: Consumer Protection Act ---
    ("What is an unfair trade practice under consumer law?", 1),
    ("What is a misleading advertisement?", 1),
    ("What are the powers of the State Commission?", 1),
    ("What is the role of the Central Consumer Protection Authority?", 1),
    ("What is the jurisdiction of the National Commission?", 1),
    ("What remedies are available to a consumer?", 1),
    ("How does the Act define a defect in goods?", 1),
    ("What is the procedure for filing a consumer complaint?", 1),
    ("What penalties exist for false advertising?", 1),
    ("Who can file a complaint under the Consumer Protection Act?", 1),
    ("What is meant by deficiency in service?", 1),
    ("What is the composition of the State Council?", 1),
    ("What is e-commerce liability under the Act?", 1),
    ("What is product liability action?", 1),
    ("What is mediation under the Consumer Protection Act?", 1),
    # --- Tricky/borderline in-domain (harder phrasing) ---
    ("If a shop refuses to refund a defective product, what can I do?", 1),
    ("Can the government restrict my freedom of speech?", 1),
    ("Is there a right to privacy under Indian law?", 1),
    ("What happens if a company sells expired goods?", 1),
    ("Can I sue for a warranty violation on an appliance?", 1),
    # --- Clearly out-of-domain (unrelated topics) ---
    ("What is the statutory penalty for driving without a license?", 0),
    ("How do I file an income tax return?", 0),
    ("What is the punishment for theft under the IPC?", 0),
    ("What are the visa requirements for traveling to the US?", 0),
    ("How do I register a trademark in India?", 0),
    ("What is the process for divorce under the Hindu Marriage Act?", 0),
    ("What is the weather like today?", 0),
    ("How do I apply for a passport?", 0),
    ("What is the GST rate on electronics?", 0),
    ("Explain the rules of cricket.", 0),
    ("What is the best programming language to learn?", 0),
    ("How do I cook biryani?", 0),
    ("What is the penalty for cheque bounce under the Negotiable Instruments Act?", 0),
    ("What are the eligibility criteria for a home loan?", 0),
    ("How is company registration done under the Companies Act?", 0),
    # --- Harder out-of-domain: legal-SOUNDING but wrong law entirely ---
    ("What is the punishment for murder under the IPC?", 0),
    ("What is the procedure for bail under CrPC?", 0),
    ("What are the grounds for anticipatory bail?", 0),
    ("What is the Right to Information Act about?", 0),
    ("What does the Motor Vehicles Act say about drunk driving?", 0),
    ("What is the minimum wage law in India?", 0),
    ("What are the labor laws for maternity leave?", 0),
    ("What is the Arbitration and Conciliation Act?", 0),
    ("What is cyberbullying law in India?", 0),
    ("What is the Environmental Protection Act about?", 0),
    ("What are the rules under the Juvenile Justice Act?", 0),
    ("What is the Prevention of Corruption Act?", 0),
    ("How does copyright law protect authors?", 0),
    ("What is the Insolvency and Bankruptcy Code?", 0),
    ("What is the Real Estate Regulation Act (RERA)?", 0),
    # --- Ambiguous/tricky phrasing (legal words but wrong domain) ---
    ("What rights do I have if arrested by police?", 0),
    ("Can my landlord evict me without notice?", 0),
    ("What is the legal age for marriage in India?", 0),
    ("What is the punishment for cybercrime?", 0),
    ("Is triple talaq illegal in India?", 0),
]

engine = LegalRAGEngine(model_name="llama3.2:1b")

y_true, y_pred = [], []
print(f"Running guardrail evaluation on {len(GUARDRAIL_TEST_SET)} questions...\n")

for i, (question, true_label) in enumerate(GUARDRAIL_TEST_SET):
    _, is_confident, min_distance, predicted_class = engine.retrieve(question)
    pred_label = 1 if is_confident else 0
    y_true.append(true_label)
    y_pred.append(pred_label)
    status = "✓" if pred_label == true_label else "✗ MISS"
    print(f"[{i+1:2d}/{len(GUARDRAIL_TEST_SET)}] {status}  true={true_label} pred={pred_label}  dist={min_distance:.3f}  | {question[:60]}")

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\n{'='*70}")
print(f"RETRIEVAL GUARDRAIL — Evaluation on {len(GUARDRAIL_TEST_SET)} questions")
print(f"{'='*70}")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-score:  {f1*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["out_of_domain", "in_domain"], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["out_of_domain", "in_domain"], yticklabels=["out_of_domain", "in_domain"])
plt.title(f"Retrieval Guardrail - Confusion Matrix (n={len(GUARDRAIL_TEST_SET)})", fontsize=13, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_guardrail_large.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [acc*100, prec*100, rec*100, f1*100]
bars = plt.bar(metric_names, metric_values, color="#c5a059")
for bar, val in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.2f}%", ha="center", fontweight="bold")
plt.ylim(0, 105)
plt.title(f"Retrieval Guardrail Performance (n={len(GUARDRAIL_TEST_SET)} questions)", fontsize=13, fontweight="bold")
plt.ylabel("Score (%)")
plt.tight_layout()
plt.savefig("outputs/evaluation_metrics_guardrail_large.png", dpi=150)
plt.close()

print("\nSaved: outputs/confusion_matrix_guardrail_large.png")
print("Saved: outputs/evaluation_metrics_guardrail_large.png")
