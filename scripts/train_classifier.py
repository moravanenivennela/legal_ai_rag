import json, pickle, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
with open('data/dataset_1000.json') as f: data = json.load(f)
df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['query'], df['domain'], test_size=0.2, random_state=42, stratify=df['domain'])
print('Encoding text using BGE-M3 model...')
model = SentenceTransformer('BAAI/bge-m3')
X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train_emb, y_train)
y_pred = clf.predict(X_test_emb)
labels = ['constitution', 'consumer_protection', 'out_of_domain']
cm = confusion_matrix(y_test, y_pred, labels=labels)
print('\n==================== CONFUSION MATRIX ====================')
print(cm)
print('\n================ CLASSIFICATION REPORT =================')
print(classification_report(y_test, y_pred, labels=labels))
np.save('outputs/confusion_matrix.npy', cm)
with open('models/domain_classifier.pkl', 'wb') as f: pickle.dump(clf, f)
print('Saved confusion matrix to outputs/ and classifier to models/')
