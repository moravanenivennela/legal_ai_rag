"""
Qualitative sample-predictions grid for the CNN image classifier.
Shows N test images with predicted vs true label, green border if correct,
red border if wrong. Requires the saved checkpoint (legal_image_classifier.pt).
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

IMG_DATA_DIR = "consumer_law_images"
CHECKPOINT_PATH = "legal_image_classifier.pt"
GRID_SIZE = 4

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
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.eval()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

n_samples = GRID_SIZE * GRID_SIZE
fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(GRID_SIZE * 3, GRID_SIZE * 3))

shown = 0
with torch.no_grad():
    for imgs, lbls in val_loader:
        if shown >= n_samples:
            break
        output = model(imgs)
        pred = torch.argmax(output, dim=1).item()
        true = lbls.item()

        img = imgs[0].permute(1, 2, 0).numpy()
        img = np.clip(img * std + mean, 0, 1)

        r, c = divmod(shown, GRID_SIZE)
        ax = axes[r][c]
        ax.imshow(img)
        ax.axis("off")
        correct = pred == true
        color = "#10b981" if correct else "#ef4444"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax.set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[true]}", fontsize=9,
                     color=color, fontweight="bold")
        shown += 1

plt.suptitle("Document Image Classifier - Sample Predictions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/sample_predictions_grid.png", dpi=150)
plt.close()

print(f"Saved: outputs/sample_predictions_grid.png ({shown} samples shown)")
