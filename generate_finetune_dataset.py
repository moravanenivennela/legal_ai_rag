import fitz
import ollama
import json
import re

PDFS = {
    "data/constitution_of_india.pdf": "Constitution of India",
    "data/consumer_protection_act_2019.pdf": "Consumer Protection Act 2019",
}

QA_PER_CHUNK = 2
training_pairs = []

for pdf_path, doc_name in PDFS.items():
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    words = full_text.split()
    chunks = [" ".join(words[i:i+250]) for i in range(0, len(words), 250)]
    chunks = [c for c in chunks if len(c.strip()) > 100]

    print(f"{doc_name}: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:80]):  # limit for reasonable dataset size
        prompt = f"""Based on this legal text from the {doc_name}, generate {QA_PER_CHUNK} question-answer pairs.
Format EXACTLY as:
Q: <question>
A: <answer>
Q: <question>
A: <answer>

TEXT:
{chunk[:1200]}
"""
        try:
            response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
            text = response['message']['content']
            qa_pairs = re.findall(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)", text, re.DOTALL)
            for q, a in qa_pairs:
                q, a = q.strip(), a.strip()
                if len(q) > 10 and len(a) > 10:
                    training_pairs.append({"instruction": q, "output": a, "source": doc_name})
            print(f"  chunk {i+1}/{min(len(chunks),80)}: +{len(qa_pairs)} pairs (total: {len(training_pairs)})")
        except Exception as e:
            print(f"  error on chunk {i}: {e}")

with open("finetune_dataset.json", "w") as f:
    json.dump(training_pairs, f, indent=2)

print(f"\nTotal training pairs generated: {len(training_pairs)}")
print("Saved to finetune_dataset.json")
