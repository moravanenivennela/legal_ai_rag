"""
Per-class accuracy bar chart for the CNN document image classifier.
Uses the same validation split as your training script so numbers match.
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

IMG_DATA_DIR = "consumer_law_images"
CHECKPOINT_PATH = "legal_image_classifier.pt"

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

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.eval()

correct_per_class = {c: 0 for c in range(len(class_names))}
total_per_class = {c: 0 for c in range(len(class_names))}

with torch.no_grad():
    for imgs, lbls in val_loader:
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        for p, l in zip(preds.tolist(), lbls.tolist()):
            total_per_class[l] += 1
            if p == l:
                correct_per_class[l] += 1

accuracies = [
    (correct_per_class[c] / total_per_class[c] * 100) if total_per_class[c] > 0 else 0.0
    for c in range(len(class_names))
]

plt.figure(figsize=(8, 5.5))
bars = plt.bar(class_names, accuracies, color="#3b82f6")
for bar, val, c in zip(bars, accuracies, range(len(class_names))):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
              f"{val:.1f}%\n(n={total_per_class[c]})", ha="center", fontsize=9, fontweight="bold")
plt.ylim(0, 110)
plt.ylabel("Accuracy (%)")
plt.title("Document Image Classifier - Per-Class Accuracy", fontsize=13, fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("outputs/per_class_accuracy_image_classifier.png", dpi=150)
plt.close()

print("Saved: outputs/per_class_accuracy_image_classifier.png")
for c, name in enumerate(class_names):
    print(f"{name}: {accuracies[c]:.2f}% (n={total_per_class[c]})")
