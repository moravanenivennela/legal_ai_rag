import ollama
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from langchain_huggingface import HuggingFaceEmbeddings

# --- Step 1: Generate additional training questions via local LLM ---
CATEGORIES = {
    "constitution": "Ask a short, specific question about the Indian Constitution (Articles, Fundamental Rights, Directive Principles, Parliament, judiciary, President, amendments, etc.)",
    "consumer_protection": "Ask a short, specific question about India's Consumer Protection Act 2019 (consumer rights, complaints, commissions, unfair trade practices, product liability, e-commerce, etc.)",
    "out_of_domain": "Ask a short, general legal or everyday question that is clearly NOT about the Indian Constitution or Consumer Protection Act (e.g. IPC, CrPC, tax law, traffic law, cooking, sports, technology, weather, other unrelated topics)"
}

QUESTIONS_PER_CATEGORY = 30
generated_data = []

for label, instruction in CATEGORIES.items():
    print(f"Generating {QUESTIONS_PER_CATEGORY} new questions for: {label}")
    seen = set()
    attempts = 0
    while len(seen) < QUESTIONS_PER_CATEGORY and attempts < QUESTIONS_PER_CATEGORY * 4:
        attempts += 1
        prompt = f"{instruction}\nReply with ONLY the question text, nothing else, no numbering, no quotes."
        try:
            response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": prompt}], options={"temperature": 1.0})
            q = response['message']['content'].strip().strip('"')
            if 10 < len(q) < 200 and q not in seen:
                seen.add(q)
        except Exception:
            time.sleep(1)
    for q in seen:
        generated_data.append((q, label))
    print(f"  -> got {len(seen)} unique questions")

# --- Step 2: Original 45 examples ---
ORIGINAL_DATA = [
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
    ("What is the statutory penalty for driving without a license?", "out_of_domain"),
    ("How do I file an income tax return?", "out_of_domain"),
    ("What is the punishment for theft under the IPC?", "out_of_domain"),
    ("What are the visa requirements for traveling to the US?", "out_of_domain"),
    ("How do I register a trademark in India?", "out_of_domain"),
    ("What is the process for divorce under the Hindu Marriage Act?", "out_of_domain"),
    ("What is the weather like today?", "out_of_domain"),
    ("How do I apply for a passport?", "out_of_domain"),
    ("What is the GST rate on electronics?", "out_of_domain"),
    ("Explain the rules of cricket.", "out_of_domain"),
    ("What is the best programming language to learn in 2026?", "out_of_domain"),
    ("How do I cook biryani?", "out_of_domain"),
    ("What is the penalty for cheque bounce under the Negotiable Instruments Act?", "out_of_domain"),
    ("What are the eligibility criteria for a home loan?", "out_of_domain"),
    ("How is company registration done under the Companies Act?", "out_of_domain"),
]

ALL_DATA = ORIGINAL_DATA + generated_data
print(f"\nTotal training examples: {len(ALL_DATA)} (was {len(ORIGINAL_DATA)}, added {len(generated_data)})")

with open("expanded_classifier_data.json", "w") as f:
    json.dump(ALL_DATA, f, indent=2)

# --- Step 3: Retrain and evaluate ---
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
texts = [t[0] for t in ALL_DATA]
labels = [t[1] for t in ALL_DATA]
X = np.array(embedder.embed_documents(texts))
y = np.array(labels)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000)
y_pred = cross_val_predict(clf, X, y, cv=cv)

acc = accuracy_score(y, y_pred); prec = precision_score(y, y_pred, average='macro')
rec = recall_score(y, y_pred, average='macro'); f1 = f1_score(y, y_pred, average='macro')
print(f"\nEXPANDED Query Classifier: Acc={acc*100:.2f}% Prec={prec*100:.2f}% Rec={rec*100:.2f}% F1={f1*100:.2f}%")

clf.fit(X, y)
with open("query_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Retrained classifier saved to query_classifier.pkl (this REPLACES your app's live model)")

class_labels = sorted(set(y))
cm = confusion_matrix(y, y_pred, labels=class_labels)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title(f"Query Classifier - Confusion Matrix (n={len(ALL_DATA)}, expanded)", fontweight="bold")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("dl_outputs/confusion_matrix_query_classifier.png")
plt.close()

plt.figure(figsize=(7,5))
names = ["Accuracy","Precision","Recall","F1 Score"]
vals = [acc*100, prec*100, rec*100, f1*100]
bars = plt.bar(names, vals, color="#3b82f6")
for bar, val in zip(bars, vals):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val:.2f}%", ha="center", fontweight="bold")
plt.ylim(0,100)
plt.title(f"Query Classifier - Model Performance (n={len(ALL_DATA)}, expanded dataset)", fontweight="bold")
plt.tight_layout()
plt.savefig("dl_outputs/evaluation_metrics_query_classifier.png")
plt.close()
print("\nUpdated charts saved to dl_outputs/")
