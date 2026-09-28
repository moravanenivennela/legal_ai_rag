from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from rag_engine import LegalRAGEngine

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

y_true, y_pred = [], []
for question, true_label in GUARDRAIL_TEST_SET:
    _, is_confident, _, _ = engine.retrieve(question)
    y_true.append(true_label)
    y_pred.append(1 if is_confident else 0)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
total = TN + FP + FN + TP

def compute_for(TP, FN, FP, TN, name):
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    lr_plus = sens / (1 - spec) if (1 - spec) > 0 else float('inf')
    lr_minus = (1 - sens) / spec if spec > 0 else float('inf')
    return [name, f"{sens*100:.1f}%", f"{spec*100:.1f}%", f"{ppv*100:.1f}%", f"{npv*100:.1f}%",
            f"{lr_plus:.2f}" if lr_plus != float('inf') else "inf",
            f"{lr_minus:.2f}" if lr_minus != float('inf') else "inf"]

row_in_domain = compute_for(TP, FN, FP, TN, "in_domain")
row_out_domain = compute_for(TN, FP, FN, TP, "out_of_domain")

print(f"Confusion Matrix (n={total}): TN={TN} FP={FP} FN={FN} TP={TP}")
print(row_in_domain)
print(row_out_domain)

fig, ax = plt.subplots(figsize=(10, 2.2))
ax.axis("off")
col_labels = ["Class", "Sensitivity", "Specificity", "PPV", "NPV", "LR+", "LR-"]
rows = [row_out_domain, row_in_domain]
table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor("#3b82f6")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
plt.title(f"Retrieval Guardrail - Diagnostic Metrics (n={total} questions)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/diagnostic_metrics_guardrail_large.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/diagnostic_metrics_guardrail_large.png")
