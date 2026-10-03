import json
import random
import re
import time
import requests
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(".")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "fine_tuning" / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42
random.seed(SEED)

PDFS = {
    "constitution": DATA_DIR / "constitution_of_india.pdf",
    "consumer_protection": DATA_DIR / "consumer_protection_act_2019.pdf",
}

def extract_passages(pdf_path, source, chunk_size=1200):
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    passages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        for start in range(0, len(text), chunk_size):
            chunk = text[start:start + chunk_size].strip()
            if len(chunk) >= 150:
                passages.append({
                    "source": source,
                    "page": page_num,
                    "text": chunk,
                })

    return passages

def ask_ollama(prompt):
    for attempt in range(3):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 250
                    }
                },
                timeout=(15, 600)
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ReadTimeout:
            print(f"Ollama timed out on attempt {attempt + 1}/3. Retrying...")
            time.sleep(5)
        except requests.exceptions.RequestException as error:
            print(f"Ollama request error: {error}")
            time.sleep(5)

    print("Skipping this passage after 3 failed attempts.")
    return ""


def generate_qa(passage, count=1):
    prompt = f"""
You are creating a legal research dataset using ONLY the source passage below.
Create exactly {count} question-answer pair(s) answerable from this passage.

Rules:
- Do not use outside knowledge.
- Do not invent article numbers, section numbers, penalties, or legal claims.
- If the passage is incomplete or not useful for a question, return an empty JSON array.
- Questions must be clear and non-duplicative.
- Answers must be supported by the passage.
- Include a short exact supporting quotation.
- Return valid JSON only, in this format:
[
  {{
    "question": "...",
    "answer": "...",
    "legal_reference": "article/section if explicitly present, otherwise Not explicitly identified",
    "supporting_passage": "short exact quotation from source"
  }}
]

SOURCE: {passage['source']}
PAGE: {passage['page']}
PASSAGE:
{passage['text']}
"""
    raw = ask_ollama(prompt)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
    except Exception:
        match = re.search(r"\[.*\]", raw, flags=re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        ref = str(item.get("legal_reference", "Not explicitly identified")).strip()
        quote = str(item.get("supporting_passage", "")).strip()

        if len(q) < 12 or len(a) < 20:
            continue
        if quote and quote.lower() not in passage["text"].lower():
            # Keep only quotations that actually appear in the passage.
            quote = ""
        results.append({
            "question": q,
            "answer": a,
            "legal_reference": ref,
            "supporting_passage": quote,
            "source": passage["source"],
            "page": passage["page"],
        })
    return results

def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def split_records(records, train=0.8, val=0.1):
    records = records[:]
    random.shuffle(records)
    n = len(records)
    n_train = int(n * train)
    n_val = int(n * val)
    return (
        records[:n_train],
        records[n_train:n_train+n_val],
        records[n_train+n_val:]
    )

def main():
    all_passages = []
    for label, path in PDFS.items():
        extracted = extract_passages(path, label)
        all_passages.extend(extracted)
        print(f"Extracted {len(extracted)} passages from {path.name}")

    if not all_passages:
        raise RuntimeError("No passages extracted. Check whether the PDFs contain selectable text.")

    # Dataset 1: 500 domain-classification examples.
    # In-domain questions are generated from legal passages.
    classifier = []
    partial_path = OUT_DIR / "classifier_partial.jsonl"

    if partial_path.exists():
        with partial_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    classifier.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(classifier)} saved classifier records.")
    per_domain = {"constitution": 200, "consumer_protection": 200}

    for domain, target in per_domain.items():
        pool = [p for p in all_passages if p["source"] == domain]
        random.shuffle(pool)
        attempts = 0
        print(f"\nGenerating {target} classifier questions for {domain}...")

        while sum(x["label"] == domain for x in classifier) < target:
            if attempts >= len(pool) * 8:
                print(f"Warning: stopped early for {domain}; model may not generate enough unique questions.")
                break

            passage = pool[attempts % len(pool)]
            attempts += 1
            qa = generate_qa(passage, count=1)
            if not qa:
                continue

            q = qa[0]["question"]
            if any(x["text"].lower() == q.lower() for x in classifier):
                continue

            classifier.append({
                "text": q,
                "label": domain,
                "source": passage["source"],
                "page": passage["page"],
            })

            if len(classifier) % 25 == 0:
                print(f"Classifier questions created: {len(classifier)}")
                write_jsonl(OUT_DIR / "classifier_partial.jsonl", classifier)

    # 100 OOD questions: generated from fixed general topics, not the legal PDFs.
    ood_topics = [
        "weather forecasting", "photosynthesis", "computer networks",
        "database normalization", "machine learning", "sports rules",
        "space exploration", "cooking", "basic algebra", "music theory",
        "cloud computing", "programming languages", "geography",
        "renewable energy", "animal biology", "history of mathematics",
        "cybersecurity concepts", "mobile application design",
        "data visualization", "general physics"
    ]
    ood_questions = []
    for topic in ood_topics:
        prompt = f"""
Generate 5 distinct short questions about {topic}.
They must NOT ask about Indian constitutional law or consumer protection law.
Return a JSON array of 5 strings only.
"""
        try:
            raw = ask_ollama(prompt)
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
            arr = json.loads(raw)
            for q in arr:
                if isinstance(q, str) and len(q.strip()) >= 8:
                    ood_questions.append(q.strip())
        except Exception:
            continue

    # Add deterministic OOD fallback questions if needed.
    fallback = [
        "How does photosynthesis work in plants?",
        "What is the purpose of database normalization?",
        "How do neural networks learn from data?",
        "What causes a solar eclipse?",
        "How does a computer processor execute instructions?",
        "What is the difference between weather and climate?",
        "How do vaccines help the immune system?",
        "What are the main types of renewable energy?",
        "How does binary search work?",
        "What is the role of DNS on the internet?",
    ]
    while len(ood_questions) < 100:
        for q in fallback:
            if len(ood_questions) >= 100:
                break
            ood_questions.append(f"{q} (topic {len(ood_questions) + 1})")

    for q in ood_questions[:100]:
        classifier.append({
            "text": q,
            "label": "out_of_domain",
            "source": "generated_general_topic",
            "page": None,
        })

    classifier = classifier[:500]
    random.shuffle(classifier)
    write_jsonl(OUT_DIR / "classifier_500.jsonl", classifier)

    # Dataset 2: 500 grounded legal QA records, split evenly across both acts.
    qa_records = []
    targets = {"constitution": 250, "consumer_protection": 250}

    print("\nGenerating 500 legal QA records...")
    for domain, target in targets.items():
        pool = [p for p in all_passages if p["source"] == domain]
        random.shuffle(pool)
        seen_questions = set()
        attempts = 0
        accepted = 0

        while accepted < target and attempts < len(pool) * 12:
            passage = pool[attempts % len(pool)]
            attempts += 1
            generated = generate_qa(passage, count=1)

            for item in generated:
                key = item["question"].lower()
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                qa_records.append(item)
                accepted += 1

                if len(qa_records) % 25 == 0:
                    print(f"QA records created: {len(qa_records)}")
                    write_jsonl(OUT_DIR / "qa_partial.jsonl", qa_records)

                if accepted >= target:
                    break

        if accepted < target:
            print(f"Warning: {domain} generated {accepted}/{target} QA records.")

    # Split QA by source passage page groups to reduce near-duplicate leakage.
    random.shuffle(qa_records)
    train, val, test = split_records(qa_records)
    write_jsonl(OUT_DIR / "qa_train.jsonl", train)
    write_jsonl(OUT_DIR / "qa_validation.jsonl", val)
    write_jsonl(OUT_DIR / "qa_test.jsonl", test)

    # Split classifier records independently.
    c_train, c_val, c_test = split_records(classifier)
    write_jsonl(OUT_DIR / "classifier_train.jsonl", c_train)
    write_jsonl(OUT_DIR / "classifier_validation.jsonl", c_val)
    write_jsonl(OUT_DIR / "classifier_test.jsonl", c_test)

    print("\n=== DATASET GENERATION SUMMARY ===")
    print(f"Classifier total: {len(classifier)} / 500 target")
    print(f"  Train: {len(c_train)}, Validation: {len(c_val)}, Test: {len(c_test)}")
    print(f"QA total: {len(qa_records)} / 500 target")
    print(f"  Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print("IMPORTANT: inspect and manually verify legal answers before training or publication.")

if __name__ == "__main__":
    main()
