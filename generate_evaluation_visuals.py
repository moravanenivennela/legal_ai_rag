import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression

os.makedirs("outputs", exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

# ============================================================
# PART A: QUERY DOMAIN CLASSIFIER
# ============================================================
log("=" * 70)
log("A) QUERY DOMAIN CLASSIFIER — Evaluation")
log("=" * 70)

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
y_pred = cross_val_predict(clf, X, y, cv=cv)

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, average='macro')
rec = recall_score(y, y_pred, average='macro')
f1 = f1_score(y, y_pred, average='macro')

log(f"\nAccuracy:  {acc*100:.2f}%")
log(f"Precision: {prec*100:.2f}%")
log(f"Recall:    {rec*100:.2f}%")
log(f"F1-score:  {f1*100:.2f}%")
log("\nClassification Report:")
log(classification_report(y, y_pred))

class_labels = sorted(set(y))
cm = confusion_matrix(y, y_pred, labels=class_labels)
log("Confusion Matrix:")
log(str(cm))

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, cbar=True)
plt.title("Query Classifier - Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_query_classifier.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [acc*100, prec*100, rec*100, f1*100]
bars = plt.bar(metric_names, metric_values, color="#3b82f6")
for bar, val in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.2f}%", ha="center", fontweight="bold")
plt.ylim(0, 100)
plt.title("Query Classifier - Model Performance", fontsize=14, fontweight="bold")
plt.ylabel("Score (%)")
plt.tight_layout()
plt.savefig("outputs/evaluation_metrics_query_classifier.png", dpi=150)
plt.close()

log("\nSaved: outputs/confusion_matrix_query_classifier.png")
log("Saved: outputs/evaluation_metrics_query_classifier.png")


# ============================================================
# PART B: DOCUMENT IMAGE CLASSIFIER (CNN) — WITH TRAINING CURVES
# ============================================================
log("\n" + "=" * 70)
log("B) DOCUMENT IMAGE CLASSIFIER (CNN) — Training & Evaluation")
log("=" * 70)

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
log(f"Classes: {class_names}, total images: {len(full_dataset)}")

torch.manual_seed(42)
val_size = max(1, int(0.2 * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 8
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, lbls in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    model.eval()
    val_loss_total, val_correct, val_total = 0, 0, 0
    all_val_preds, all_val_true = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            val_loss_total += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == lbls).sum().item()
            val_total += lbls.size(0)
            all_val_preds.extend(preds.tolist())
            all_val_true.extend(lbls.tolist())
    val_loss = val_loss_total / len(val_loader)
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    log(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

torch.save({"model_state": model.state_dict(), "classes": class_names}, "legal_image_classifier.pt")

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Training Loss", color="#3b82f6")
plt.plot(history["val_loss"], label="Validation Loss", color="#f59e0b")
plt.title("CNN Training and Validation Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/cnn_training_loss.png", dpi=150)
plt.close()

# Accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(history["train_acc"], label="Training Accuracy", color="#3b82f6")
plt.plot(history["val_acc"], label="Validation Accuracy", color="#f59e0b")
plt.title("CNN Training and Validation Accuracy", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/cnn_training_accuracy.png", dpi=150)
plt.close()

img_acc = accuracy_score(all_val_true, all_val_preds)
img_prec = precision_score(all_val_true, all_val_preds, average='macro', zero_division=0)
img_rec = recall_score(all_val_true, all_val_preds, average='macro', zero_division=0)
img_f1 = f1_score(all_val_true, all_val_preds, average='macro', zero_division=0)

log(f"\nFinal Validation Accuracy:  {img_acc*100:.2f}%")
log(f"Final Validation Precision: {img_prec*100:.2f}%")
log(f"Final Validation Recall:    {img_rec*100:.2f}%")
log(f"Final Validation F1-score:  {img_f1*100:.2f}%")
log("\nClassification Report:")
log(classification_report(all_val_true, all_val_preds, target_names=class_names, zero_division=0))

cm_img = confusion_matrix(all_val_true, all_val_preds)
log("Confusion Matrix:")
log(str(cm_img))

plt.figure(figsize=(7, 6))
sns.heatmap(cm_img, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title("Document Image Classifier - Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_image_classifier.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
metric_values_img = [img_acc*100, img_prec*100, img_rec*100, img_f1*100]
bars = plt.bar(metric_names, metric_values_img, color="#10b981")
for bar, val in zip(bars, metric_values_img):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.2f}%", ha="center", fontweight="bold")
plt.ylim(0, 100)
plt.title("Document Image Classifier - Model Performance", fontsize=14, fontweight="bold")
plt.ylabel("Score (%)")
plt.tight_layout()
plt.savefig("outputs/evaluation_metrics_image_classifier.png", dpi=150)
plt.close()

log("\nSaved: outputs/cnn_training_loss.png")
log("Saved: outputs/cnn_training_accuracy.png")
log("Saved: outputs/confusion_matrix_image_classifier.png")
log("Saved: outputs/evaluation_metrics_image_classifier.png")

log("\n" + "=" * 70)
log("Evaluation completed successfully! All charts saved in ./outputs/")
log("=" * 70)

with open("outputs/evaluation_results.txt", "w") as f:
    f.write("\n".join(log_lines))
