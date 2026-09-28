"""
Retrains the query domain classifier with an EXPANDED dataset (more examples
per class + hard negatives) and evaluates with cross-validation to get a
realistic, honest accuracy estimate. Also saves the newly trained classifier
so rag_engine.py picks it up automatically.
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# EXPANDED TRAINING DATA
# Original set + additional examples + HARD NEGATIVES (out-of-domain
# questions that use legal-sounding words, which are the usual source
# of classifier confusion).
# ============================================================
TRAINING_DATA = [
    # --- consumer_protection ---
    ("What are the six consumer rights under the Consumer Protection Act?", "consumer_protection"),
    ("What is the pecuniary jurisdiction of the District Commission?", "consumer_protection"),
    ("Explain Product Liability under the 2019 Act.", "consumer_protection"),
    ("What is an unfair trade practice?", "consumer_protection"),
    ("What is a misleading advertisement under consumer law?", "consumer_protection"),
    ("What are the powers of the State Commission?", "consumer_protection"),
    ("What is the role of the Central Consumer Protection Authority?", "consumer_protection"),
    ("What is the jurisdiction of the National Commission?", "consumer_protection"),
    ("What remedies are available to a consumer under this Act?", "consumer_protection"),
    ("How does the Act define a defect in goods?", "consumer_protection"),
    ("What is the procedure for filing a consumer complaint?", "consumer_protection"),
    ("What penalties exist for false advertising under consumer law?", "consumer_protection"),
    ("Who can file a complaint under the Consumer Protection Act?", "consumer_protection"),
    ("What is meant by deficiency in service?", "consumer_protection"),
    ("What is the composition of the State Council?", "consumer_protection"),
    ("What is e-commerce liability under the Act?", "consumer_protection"),
    ("What is product liability action?", "consumer_protection"),
    ("What is mediation under the Consumer Protection Act?", "consumer_protection"),
    ("If a shop refuses to refund a defective product, what can I do?", "consumer_protection"),
    ("What happens if a company sells expired goods?", "consumer_protection"),
    ("Can I sue for a warranty violation on an appliance?", "consumer_protection"),
    ("What is the limitation period for filing a consumer complaint?", "consumer_protection"),
    ("What compensation can I claim for a defective product?", "consumer_protection"),
    ("What is the difference between goods and services under the Act?", "consumer_protection"),
    ("How does the Act protect against spurious goods?", "consumer_protection"),
    ("What is a class action complaint under consumer law?", "consumer_protection"),
    ("What is the appeal process from District to State Commission?", "consumer_protection"),
    ("What are unfair contract terms under the Act?", "consumer_protection"),

    # --- constitution ---
    ("What does Article 21 of the Constitution protect?", "constitution"),
    ("Explain the writ jurisdiction under Article 32.", "constitution"),
    ("How does Article 246 distribute legislative powers?", "constitution"),
    ("What are the Directive Principles of State Policy?", "constitution"),
    ("What is the scope of Article 14 on equality?", "constitution"),
    ("Explain the difference between Article 32 and Article 226.", "constitution"),
    ("What fundamental duties are listed in the Constitution?", "constitution"),
    ("What is the procedure to amend the Constitution?", "constitution"),
    ("What powers does the President have under the Constitution?", "constitution"),
    ("What is the significance of the Preamble?", "constitution"),
    ("How is the Union List different from the State List?", "constitution"),
    ("What are the Fundamental Rights guaranteed under Part III?", "constitution"),
    ("Explain the concept of judicial review in the Constitution.", "constitution"),
    ("What is Article 19 and what freedoms does it guarantee?", "constitution"),
    ("How is the Supreme Court jurisdiction defined in the Constitution?", "constitution"),
    ("What is the role of the Governor under the Constitution?", "constitution"),
    ("What is the federal structure created by the Constitution?", "constitution"),
    ("Can the government restrict my freedom of speech?", "constitution"),
    ("Is there a right to privacy under Indian law?", "constitution"),
    ("What is the basic structure doctrine?", "constitution"),
    ("What is Article 356 and President's Rule?", "constitution"),
    ("What is the difference between Fundamental Rights and Directive Principles?", "constitution"),
    ("What is the significance of Article 370?", "constitution"),
    ("How are Fundamental Rights enforced against private individuals?", "constitution"),
    ("What is the concept of separation of powers in the Constitution?", "constitution"),
    ("What is the emergency provision under Article 352?", "constitution"),
    ("What is the role of the Attorney General under the Constitution?", "constitution"),

    ("Can the police search my house without a warrant?", "constitution"),
    ("Do I have the right to a fair trial?", "constitution"),
    ("Can I be detained without being told the charges against me?", "constitution"),
    ("Is discrimination based on religion allowed by law?", "constitution"),
    ("Can the government take away my citizenship?", "constitution"),
    ("Do children have a right to free education?", "constitution"),
    ("Can I move freely to any state in India?", "constitution"),
    ("What stops the government from censoring the press?", "constitution"),
    ("Is there a right to protest peacefully?", "constitution"),
    ("Can a law be struck down by courts?", "constitution"),
    ("What happens if a state government misuses its power?", "constitution"),
    ("Do I have the right to practice any religion I choose?", "constitution"),
    ("Can Parliament pass any law it wants?", "constitution"),
    ("What protects minorities under Indian law?", "constitution"),
    # --- out_of_domain (general, clearly unrelated) ---
    ("What is the weather like today?", "out_of_domain"),
    ("How do I apply for a passport?", "out_of_domain"),
    ("What is the GST rate on electronics?", "out_of_domain"),
    ("Explain the rules of cricket.", "out_of_domain"),
    ("What is the best programming language to learn in 2026?", "out_of_domain"),
    ("How do I cook biryani?", "out_of_domain"),
    ("What are the eligibility criteria for a home loan?", "out_of_domain"),
    ("What is the capital of France?", "out_of_domain"),
    ("How do I fix a flat tire?", "out_of_domain"),
    ("What is the plot of Romeo and Juliet?", "out_of_domain"),
    ("How do I train for a marathon?", "out_of_domain"),
    ("What's a good recipe for pasta?", "out_of_domain"),

    # --- out_of_domain HARD NEGATIVES (legal-sounding but different law areas) ---
    ("What is the statutory penalty for driving without a license?", "out_of_domain"),
    ("How do I file an income tax return?", "out_of_domain"),
    ("What is the punishment for theft under the IPC?", "out_of_domain"),
    ("What are the visa requirements for traveling to the US?", "out_of_domain"),
    ("How do I register a trademark in India?", "out_of_domain"),
    ("What is the process for divorce under the Hindu Marriage Act?", "out_of_domain"),
    ("What is the penalty for cheque bounce under the Negotiable Instruments Act?", "out_of_domain"),
    ("How is company registration done under the Companies Act?", "out_of_domain"),
    ("What is the punishment for murder under the IPC?", "out_of_domain"),
    ("What is the procedure for bail under CrPC?", "out_of_domain"),
    ("What are the grounds for anticipatory bail?", "out_of_domain"),
    ("What is the Right to Information Act about?", "out_of_domain"),
    ("What does the Motor Vehicles Act say about drunk driving?", "out_of_domain"),
    ("What is the minimum wage law in India?", "out_of_domain"),
    ("What are the labor laws for maternity leave?", "out_of_domain"),
    ("What is the Arbitration and Conciliation Act?", "out_of_domain"),
    ("What is cyberbullying law in India?", "out_of_domain"),
    ("What is the Environmental Protection Act about?", "out_of_domain"),
    ("What are the rules under the Juvenile Justice Act?", "out_of_domain"),
    ("What is the Prevention of Corruption Act?", "out_of_domain"),
    ("How does copyright law protect authors?", "out_of_domain"),
    ("What is the Insolvency and Bankruptcy Code?", "out_of_domain"),
    ("What is the Real Estate Regulation Act (RERA)?", "out_of_domain"),
    ("What rights do I have if arrested by police?", "out_of_domain"),
    ("Can my landlord evict me without notice?", "out_of_domain"),
    ("What is the legal age for marriage in India?", "out_of_domain"),
    ("What is the punishment for cybercrime?", "out_of_domain"),
    ("Is triple talaq illegal in India?", "out_of_domain"),
    ("What is the process to get a legal heir certificate?", "out_of_domain"),
    ("What is the Indian Penal Code section for defamation?", "out_of_domain"),
]

print(f"Total training examples: {len(TRAINING_DATA)}")
class_counts = {}
for _, label in TRAINING_DATA:
    class_counts[label] = class_counts.get(label, 0) + 1
print("Class distribution:", class_counts)

texts = [t[0] for t in TRAINING_DATA]
labels = [t[1] for t in TRAINING_DATA]

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
print("\nEmbedding training data...")
X = np.array(embedder.embed_documents(texts))
y = np.array(labels)

# ---- Hyperparameter search for best regularization strength ----
param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=cv, scoring="accuracy")
grid.fit(X, y)
best_C = grid.best_params_["C"]
print(f"\nBest C (regularization): {best_C}")

# ---- Honest cross-validated evaluation with best hyperparameters ----
clf = LogisticRegression(max_iter=1000, C=best_C)
y_pred = cross_val_predict(clf, X, y, cv=cv)

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, average="macro")
rec = recall_score(y, y_pred, average="macro")
f1 = f1_score(y, y_pred, average="macro")

print(f"\n{'='*60}")
print("CROSS-VALIDATED PERFORMANCE (honest estimate on held-out folds)")
print(f"{'='*60}")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-score:  {f1*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y, y_pred))

if acc < 0.95:
    print("\n⚠️  Still below 95%. Look at the classification report above —")
    print("    whichever class has the lowest recall/precision needs MORE")
    print("    real training examples in that category, not just more hard")
    print("    negatives. Add 10-15 more genuinely tricky examples for that")
    print("    class and re-run this script.")
else:
    print("\n✅ Cross-validated accuracy is above 95% — this is a genuine,")
    print("   non-overfit estimate since predictions came from held-out folds.")

# ---- Fit final classifier on ALL data and save for production use ----
final_clf = LogisticRegression(max_iter=1000, C=best_C)
final_clf.fit(X, y)

with open("query_classifier.pkl", "wb") as f:
    pickle.dump(final_clf, f)

print("\nSaved trained classifier to: query_classifier.pkl")
print("(This is the file rag_engine.py loads automatically — restart your")
print(" Streamlit app or re-instantiate LegalRAGEngine to use the new model.)")
