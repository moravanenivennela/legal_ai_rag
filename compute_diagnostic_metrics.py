import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression

def diagnostic_metrics_per_class(y_true, y_pred, class_labels):
    """Computes Sensitivity, Specificity, PPV, NPV, LR+, LR- per class (one-vs-rest)."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    results = {}
    total = cm.sum()
    for i, cls in enumerate(class_labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        lr_plus = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')
        lr_minus = (1 - sensitivity) / specificity if specificity > 0 else float('inf')

        results[cls] = {
            "Sensitivity": sensitivity, "Specificity": specificity,
            "PPV": ppv, "NPV": npv, "LR+": lr_plus, "LR-": lr_minus
        }
    return results


def print_and_return_table(results, title):
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    header = f"{'Class':<22}{'Sensitivity':<13}{'Specificity':<13}{'PPV':<10}{'NPV':<10}{'LR+':<10}{'LR-':<10}"
    print(header)
    print("-" * 90)
    rows = []
    for cls, m in results.items():
        row = f"{cls:<22}{m['Sensitivity']*100:<12.1f}%{m['Specificity']*100:<12.1f}%{m['PPV']*100:<9.1f}%{m['NPV']*100:<9.1f}%{m['LR+']:<10.2f}{m['LR-']:<10.2f}"
        print(row)
        rows.append([cls, f"{m['Sensitivity']*100:.1f}%", f"{m['Specificity']*100:.1f}%",
                     f"{m['PPV']*100:.1f}%", f"{m['NPV']*100:.1f}%", f"{m['LR+']:.2f}", f"{m['LR-']:.2f}"])
    return rows


def save_table_image(rows, title, filename):
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.2))
    ax.axis("off")
    col_labels = ["Class", "Sensitivity", "Specificity", "PPV", "NPV", "LR+", "LR-"]
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#3b82f6")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved table image: {filename}")


# ============================================================
# A) QUERY DOMAIN CLASSIFIER
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

class_labels_query = sorted(set(y))
results_query = diagnostic_metrics_per_class(y, y_pred, class_labels_query)
rows_query = print_and_return_table(results_query, "A) QUERY DOMAIN CLASSIFIER - Diagnostic Metrics")
save_table_image(rows_query, "Query Domain Classifier - Diagnostic Metrics", "outputs/diagnostic_metrics_query_classifier.png")


# ============================================================
# B) DOCUMENT IMAGE CLASSIFIER (CNN)
# ============================================================
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

torch.manual_seed(42)
val_size = max(1, int(0.2 * len(full_dataset)))
train_size = len(full_dataset) - val_size
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

class_indices = list(range(len(class_names)))
results_cnn = diagnostic_metrics_per_class(all_true, all_preds, class_indices)
results_cnn_named = {class_names[k]: v for k, v in results_cnn.items()}
rows_cnn = print_and_return_table(results_cnn_named, "B) DOCUMENT IMAGE CLASSIFIER (CNN) - Diagnostic Metrics")
save_table_image(rows_cnn, "Document Image Classifier - Diagnostic Metrics", "outputs/diagnostic_metrics_image_classifier.png")


# ============================================================
# C) RETRIEVAL GUARDRAIL (Binary)
# ============================================================
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

results_guard = diagnostic_metrics_per_class(y_true_guard, y_pred_guard, [0, 1])
results_guard_named = {"out_of_domain" if k == 0 else "in_domain": v for k, v in results_guard.items()}
rows_guard = print_and_return_table(results_guard_named, "C) RETRIEVAL GUARDRAIL - Diagnostic Metrics")
save_table_image(rows_guard, "Retrieval Guardrail - Diagnostic Metrics", "outputs/diagnostic_metrics_guardrail.png")

print("\nAll diagnostic metric tables saved to ./outputs/")
