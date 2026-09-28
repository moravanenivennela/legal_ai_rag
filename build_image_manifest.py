import os
import json

OUT_DIR = "consumer_law_images"
manifest = {}

for category in os.listdir(OUT_DIR):
    cat_path = os.path.join(OUT_DIR, category)
    if os.path.isdir(cat_path):
        images = [os.path.join(cat_path, f) for f in os.listdir(cat_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        manifest[category] = images
        print(f"{category}: {len(images)} images")

with open("class_all_images.json", "w") as f:
    json.dump(manifest, f)

print("Saved manifest to class_all_images.json")
