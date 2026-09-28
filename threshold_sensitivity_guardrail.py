"""
Threshold sensitivity plot: shows Accuracy / Precision / Recall / F1 as the
guardrail's DISTANCE threshold varies, and marks your actual chosen threshold
(DISTANCE_GUARDRAIL_THRESHOLD = 1.35 in rag_engine.py).
Note: predicting "confident/in-domain" here means min_distance <= threshold,
which is the opposite direction of a similarity score.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rag_engine import LegalRAGEngine, DISTANCE_GUARDRAIL_THRESHOLD

GUARDRAIL_TEST_SET = [
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
    ("If a shop refuses to refund a defective product, what can I do?", 1),
    ("Can the government restrict my freedom of speech?", 1),
    ("Is there a right to privacy under Indian law?", 1),
    ("What happens if a company sells expired goods?", 1),
    ("Can I sue for a warranty violation on an appliance?", 1),
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
    ("What rights do I have if arrested by police?", 0),
    ("Can my landlord evict me without notice?", 0),
    ("What is the legal age for marriage in India?", 0),
    ("What is the punishment for cybercrime?", 0),
    ("Is triple talaq illegal in India?", 0),
]

engine = LegalRAGEngine(model_name="llama3.2:1b")

y_true, distances = [], []
for question, true_label in GUARDRAIL_TEST_SET:
    _, is_confident, min_distance, predicted_class = engine.retrieve(question)
    y_true.append(true_label)
    distances.append(min_distance)

y_true = np.array(y_true)
distances = np.array(distances)

lo, hi = distances.min(), distances.max()
thresholds = np.linspace(max(0, lo - 0.1), hi + 0.1, 60)
accs, precs, recs, f1s = [], [], [], []

for t in thresholds:
    y_pred = (distances <= t).astype(int)
    accs.append(accuracy_score(y_true, y_pred))
    precs.append(precision_score(y_true, y_pred, zero_division=0))
    recs.append(recall_score(y_true, y_pred, zero_division=0))
    f1s.append(f1_score(y_true, y_pred, zero_division=0))

plt.figure(figsize=(8, 5.5))
plt.plot(thresholds, accs, label="Accuracy", color="#3b82f6")
plt.plot(thresholds, precs, label="Precision", color="#f59e0b")
plt.plot(thresholds, recs, label="Recall", color="#10b981")
plt.plot(thresholds, f1s, label="F1 Score", color="#ef4444")
plt.axvline(DISTANCE_GUARDRAIL_THRESHOLD, color="gray", linestyle="--",
            label=f"Chosen threshold ({DISTANCE_GUARDRAIL_THRESHOLD})")
plt.xlabel("Distance Threshold")
plt.ylabel("Score")
plt.title("Guardrail Threshold Sensitivity", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/threshold_sensitivity_guardrail.png", dpi=150)
plt.close()

print("Saved: outputs/threshold_sensitivity_guardrail.png")
