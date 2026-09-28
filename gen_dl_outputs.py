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

plt.rcParams['savefig.dpi'] = 300

# ============================================================
# A) Query Domain Classifier
# ============================================================
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

acc = accuracy_score(y, y_pred); prec = precision_score(y, y_pred, average='macro')
rec = recall_score(y, y_pred, average='macro'); f1 = f1_score(y, y_pred, average='macro')
print(f"Query Classifier: Acc={acc*100:.2f}% Prec={prec*100:.2f}% Rec={rec*100:.2f}% F1={f1*100:.2f}%")

class_labels = sorted(set(y))
cm = confusion_matrix(y, y_pred, labels=class_labels)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Query Classifier - Confusion Matrix", fontweight="bold")
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
plt.ylim(0,100); plt.title("Query Classifier - Model Performance", fontweight="bold")
plt.tight_layout()
plt.savefig("dl_outputs/evaluation_metrics_query_classifier.png")
plt.close()
print("Saved Query Classifier charts to dl_outputs/")

# ============================================================
# B) CNN Image Classifier — training/validation curves
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

IMG_DATA_DIR = "consumer_law_images"
transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
full_dataset = datasets.ImageFolder(IMG_DATA_DIR, transform=transform)
class_names = full_dataset.classes
torch.manual_seed(42)
val_size = max(1, int(0.2*len(full_dataset)))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
for p in model.features.parameters(): p.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 8
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
all_val_preds, all_val_true = [], []

for epoch in range(EPOCHS):
    model.train()
    tot_loss, correct, total = 0,0,0
    for imgs, lbls in train_loader:
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        tot_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        correct += (preds==lbls).sum().item(); total += lbls.size(0)
    history["train_loss"].append(tot_loss/len(train_loader))
    history["train_acc"].append(correct/total)

    model.eval()
    vloss, vcorrect, vtotal = 0,0,0
    all_val_preds, all_val_true = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            out = model(imgs)
            loss = criterion(out, lbls)
            vloss += loss.item()
            preds = torch.argmax(out, dim=1)
            vcorrect += (preds==lbls).sum().item(); vtotal += lbls.size(0)
            all_val_preds.extend(preds.tolist()); all_val_true.extend(lbls.tolist())
    history["val_loss"].append(vloss/len(val_loader))
    history["val_acc"].append(vcorrect/vtotal)
    print(f"Epoch {epoch+1}/{EPOCHS} train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}")

plt.figure(figsize=(8,5))
plt.plot(history["train_loss"], label="Training Loss", color="#3b82f6")
plt.plot(history["val_loss"], label="Validation Loss", color="#f59e0b")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss", fontweight="bold")
plt.legend(); plt.tight_layout()
plt.savefig("dl_outputs/cnn_training_loss.png")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history["train_acc"], label="Training Accuracy", color="#3b82f6")
plt.plot(history["val_acc"], label="Validation Accuracy", color="#f59e0b")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy", fontweight="bold")
plt.legend(); plt.tight_layout()
plt.savefig("dl_outputs/cnn_training_accuracy.png")
plt.close()

img_acc = accuracy_score(all_val_true, all_val_preds)
img_prec = precision_score(all_val_true, all_val_preds, average='macro', zero_division=0)
img_rec = recall_score(all_val_true, all_val_preds, average='macro', zero_division=0)
img_f1 = f1_score(all_val_true, all_val_preds, average='macro', zero_division=0)
print(f"\nCNN Final: Acc={img_acc*100:.2f}% Prec={img_prec*100:.2f}% Rec={img_rec*100:.2f}% F1={img_f1*100:.2f}%")

cm_img = confusion_matrix(all_val_true, all_val_preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm_img, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("CNN Image Classifier - Confusion Matrix", fontweight="bold")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("dl_outputs/confusion_matrix_image_classifier.png")
plt.close()

plt.figure(figsize=(7,5))
vals = [img_acc*100, img_prec*100, img_rec*100, img_f1*100]
bars = plt.bar(names, vals, color="#10b981")
for bar, val in zip(bars, vals):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val:.2f}%", ha="center", fontweight="bold")
plt.ylim(0,100); plt.title("CNN Image Classifier - Model Performance", fontweight="bold")
plt.tight_layout()
plt.savefig("dl_outputs/evaluation_metrics_image_classifier.png")
plt.close()

print("\nAll DL algorithm charts saved to dl_outputs/")
