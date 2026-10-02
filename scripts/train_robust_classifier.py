import json, pickle, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# 1. Load Data
with open("data/robust_dataset_600.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. Train/Test Split (100 Train, 500 Test to get >500 count in Confusion Matrix)
X_train, X_test, y_train, y_test = train_test_split(
    df['query'], df['domain'], test_size=500, random_state=42, stratify=df['domain']
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 3. Dense Embedding Extraction
print("Encoding texts using BGE-M3...")
model = SentenceTransformer('BAAI/bge-m3')
X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)

# 4. Logistic Regression Classifier with Regularization
clf = LogisticRegression(max_iter=1000, C=2.0)
clf.fit(X_train_emb, y_train)

# 5. Predict and Evaluate
y_pred = clf.predict(X_test_emb)
labels = ["constitution", "consumer_protection", "out_of_domain"]

cm = confusion_matrix(y_test, y_pred, labels=labels)
report = classification_report(y_test, y_pred, labels=labels, digits=4)

print("\n================ CONFUSION MATRIX (500 TEST SAMPLES) ================")
print(cm)
print(f"Total samples evaluated: {cm.sum()}")
print("\n================ CLASSIFICATION REPORT ================")
print(report)

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.save("outputs/confusion_matrix_500.npy", cm)
with open("models/robust_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Saved confusion matrix data and model artifact successfully.")
