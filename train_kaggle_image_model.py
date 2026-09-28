import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

DATA_DIR = "consumer_law_images"
MODEL_OUT = "legal_image_classifier.pt"

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}, total images: {len(full_dataset)}")

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 - loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = torch.argmax(model(imgs), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Validation accuracy: {100*correct/total:.1f}%")

    torch.save({"model_state": model.state_dict(), "classes": class_names}, MODEL_OUT)

    import os, json
    best_per_class = {}
    for cls in class_names:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            best_per_class[cls] = os.path.join(cls_dir, files[0])
    with open("class_sample_images.json", "w") as f:
        json.dump(best_per_class, f)
    print("Saved representative image per class to class_sample_images.json")

if __name__ == "__main__":
    train()
