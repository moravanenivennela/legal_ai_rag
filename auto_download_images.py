import requests
import os

PEXELS_API_KEY = "jOcrYSf3AlZax4sR8iWDN1A7SUAJoAFbRfurNnvFJnobh68g23gtgK3o"

CATEGORIES = {
    "government_building": "parliament building government",
    "courtroom": "courtroom judge gavel",
    "shopping_consumer": "shopping mall retail store",
    "contract_signing": "signing contract handshake"
}

IMAGES_PER_CATEGORY = 80  # Pexels max per request
OUT_DIR = "consumer_law_images"

def download():
    headers = {"Authorization": PEXELS_API_KEY}
    for folder, query in CATEGORIES.items():
        out_path = os.path.join(OUT_DIR, folder)
        os.makedirs(out_path, exist_ok=True)
        url = f"https://api.pexels.com/v1/search?query={query}&per_page={IMAGES_PER_CATEGORY}"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        photos = data.get("photos", [])
        print(f"{folder}: found {len(photos)} images")
        for i, photo in enumerate(photos):
            img_url = photo["src"]["medium"]
            img_data = requests.get(img_url).content
            with open(os.path.join(out_path, f"{folder}_{i+1}.jpg"), "wb") as f:
                f.write(img_data)
        print(f"  Downloaded {len(photos)} images to {out_path}")

if __name__ == "__main__":
    download()
