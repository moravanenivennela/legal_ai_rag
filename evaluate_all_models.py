"""
Formal evaluation report: Accuracy, Precision, Recall, F1-score
for all three trained/self-trained classifiers in the pipeline:
  A) Query Domain Classifier (Logistic Regression)
  B) Document Image Classifier (CNN / MobileNetV2)
  C) Retrieval Guardrail (in-domain vs out-of-domain decision)
"""
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression

print("=" * 70)
print("A) QUERY DOMAIN CLASSIFIER — Precision / Recall / F1 / Accuracy")
print("=" * 70)

from langchain_huggingface import HuggingFaceEmbeddings

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

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
texts = [t[0] for t in TRAINING_DATA]
labels = [t[1] for t in TRAINING_DATA]

X = np.array(embedder.embed_documents(texts))
y = np.array(labels)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000)
y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

print(f"\nAccuracy:  {accuracy_score(y, y_pred_cv):.3f}")
print(f"Precision (macro): {precision_score(y, y_pred_cv, average='macro'):.3f}")
print(f"Recall (macro):    {recall_score(y, y_pred_cv, average='macro'):.3f}")
print(f"F1-score (macro):  {f1_score(y, y_pred_cv, average='macro'):.3f}")
print("\nPer-class report:")
print(classification_report(y, y_pred_cv))
print("Confusion Matrix (rows=true, cols=predicted):")
labels_order = sorted(set(y))
print("Labels order:", labels_order)
print(confusion_matrix(y, y_pred_cv, labels=labels_order))


print("\n" + "=" * 70)
print("B) DOCUMENT IMAGE CLASSIFIER (CNN) — Precision / Recall / F1 / Accuracy")
print("=" * 70)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

IMG_DATA_DIR = "consumer_law_images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(IMG_DATA_DIR, transform=transform)
class_names = full_dataset.classes

val_size = max(1, int(0.2 * len(full_dataset)))
train_size = len(full_dataset) - val_size
torch.manual_seed(42)
_, val_ds = random_split(full_dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size=8)

checkpoint = torch.load("legal_image_classifier.pt", map_location="cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state"])
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for imgs, lbls in val_loader:
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_true.extend(lbls.tolist())

print(f"\nAccuracy:  {accuracy_score(all_true, all_preds):.3f}")
print(f"Precision (macro): {precision_score(all_true, all_preds, average='macro', zero_division=0):.3f}")
print(f"Recall (macro):    {recall_score(all_true, all_preds, average='macro', zero_division=0):.3f}")
print(f"F1-score (macro):  {f1_score(all_true, all_preds, average='macro', zero_division=0):.3f}")
print("\nPer-class report:")
print(classification_report(all_true, all_preds, target_names=checkpoint["classes"], zero_division=0))


print("\n" + "=" * 70)
print("C) RETRIEVAL GUARDRAIL — In-domain vs Out-of-domain Decision")
print("=" * 70)

from rag_engine import LegalRAGEngine

GUARDRAIL_TEST_SET = [
    ("What are the consumer rights under the Consumer Protection Act?", 1),
    ("What is the pecuniary jurisdiction of the District Commission?", 1),
    ("What is Product Liability under the 2019 Act?", 1),
    ("What does Article 21 of the Constitution protect?", 1),
    ("Explain the writ jurisdiction under Article 32.", 1),
    ("How does Article 246 distribute legislative powers?", 1),
    ("What are the Directive Principles of State Policy?", 1),
    ("What is the significance of the Preamble?", 1),
    ("What is the statutory penalty for driving without a license?", 0),
    ("How do I file an income tax return?", 0),
    ("What is the punishment for theft under the IPC?", 0),
    ("What are the visa requirements for traveling to the US?", 0),
    ("What is the weather like today?", 0),
    ("How do I cook biryani?", 0),
    ("What is the best programming language to learn?", 0),
    ("How is company registration done under the Companies Act?", 0),
]

engine = LegalRAGEngine(model_name="llama3.2:1b")

y_true_guard, y_pred_guard = [], []
for question, true_label in GUARDRAIL_TEST_SET:
    _, is_confident, _, _ = engine.retrieve(question)
    y_true_guard.append(true_label)
    y_pred_guard.append(1 if is_confident else 0)

print(f"\nAccuracy:  {accuracy_score(y_true_guard, y_pred_guard):.3f}")
print(f"Precision: {precision_score(y_true_guard, y_pred_guard, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_true_guard, y_pred_guard, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_true_guard, y_pred_guard, zero_division=0):.3f}")
print("\n1 = in-domain (should answer), 0 = out-of-domain (should reject)")
print(classification_report(y_true_guard, y_pred_guard, target_names=["out_of_domain", "in_domain"], zero_division=0))
