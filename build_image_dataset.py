import fitz
import os

PDF_LABELS = {
    "data/constitution_of_india.pdf": "constitution",
    "data/consumer_protection_act_2019.pdf": "consumer_protection",
}

OUTPUT_DIR = "image_dataset"

def build():
    for pdf_path, label in PDF_LABELS.items():
        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        print(f"Rendering {len(doc)} pages from {pdf_path} as '{label}'...")
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=120)
            out_path = os.path.join(out_dir, f"page_{i+1}.png")
            pix.save(out_path)
        doc.close()
    print("Dataset built at ./image_dataset/<label>/*.png")

if __name__ == "__main__":
    build()
