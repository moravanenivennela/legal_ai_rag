import numpy as np
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

TRAINING_DATA = [
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

def train():
    print(f"Total training examples: {len(TRAINING_DATA)}")
    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    texts = [t[0] for t in TRAINING_DATA]
    labels = [t[1] for t in TRAINING_DATA]

    print("Generating embeddings for training data...")
    X = np.array(embedder.embed_documents(texts))
    y = np.array(labels)

    print("Running 5-fold cross-validation for a reliable accuracy estimate...")
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"Cross-validated accuracy: {scores.mean()*100:.1f} percent")
    print(scores)

    print("Training final model on all data...")
    clf.fit(X, y)

    with open("query_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Saved final classifier to query_classifier.pkl")

if __name__ == "__main__":
    train()
