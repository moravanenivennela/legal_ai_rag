import csv
import json
import random
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(".")
MODEL = "qwen2.5:3b"
OUT = Path("fine_tuning/dataset/expanded/domain_guardrail_500")
OUT.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUT / "additional_400_candidates.jsonl"
SEED = 20261003
random.seed(SEED)

PASSAGE_FILES = [
    Path("fine_tuning/dataset/passage_backup/train_passages.jsonl"),
    Path("fine_tuning/dataset/passage_backup/validation_passages.jsonl"),
    Path("fine_tuning/dataset/passage_backup/test_passages.jsonl"),
]

CLASSIFICATION_FILES = [
    Path(f"fine_tuning/dataset/expanded/classification/repaired_{version}/{split}_classification.jsonl")
    for version in ["v2", "v3"]
    for split in ["train", "validation", "test"]
]

BLIND_CSV = Path("outputs/FINAL_BLIND_EVALUATION_100.csv")


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ollama_questions(batch, label):
    passage_text = []
    for item in batch:
        passage_text.append({
            "id": item["id"],
            "page": item["page"],
            "context": item["context"][:1000],
        })

    prompt = f"""
Create two distinct, natural user questions for EACH supplied legal passage.
All questions must genuinely relate to the supplied passage.
The intended domain label for every question is "{label}".

Requirements:
- Questions must be answerable or meaningfully related to the passage.
- Use natural wording a person might type into a legal information assistant.
- Vary wording and question structure.
- Do not invent legal facts.
- Do not simply ask "What does this passage say?"
- Avoid repeating the source name in every question.
- Return valid JSON only, with this exact structure:
{{"items":[{{"id":"P001","questions":["question one?","question two?"]}}]}}

Passages:
{json.dumps(passage_text, ensure_ascii=False)}
"""
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.25},
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    return json.loads(result["response"])


def get_existing_questions():
    existing = set()

    if BLIND_CSV.exists():
        with BLIND_CSV.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                existing.add(normalize(row.get("Question", "")))

    for path in CLASSIFICATION_FILES:
        for row in read_jsonl(path):
            existing.add(normalize(row.get("text", row.get("question", ""))))

    return {q for q in existing if q}


def get_passages():
    passages = []
    seen = set()

    for path in PASSAGE_FILES:
        for row in read_jsonl(path):
            source = str(row.get("source", "")).lower().strip()
            context = str(row.get("context", "")).strip()
            if source not in {"constitution", "consumer_protection"} or len(context) < 120:
                continue

            key = (source, normalize(context))
            if key in seen:
                continue
            seen.add(key)

            passages.append({
                "source": source,
                "page": row.get("page"),
                "chunk_id": row.get("chunk_id"),
                "context": context,
            })

    return passages


# Diverse out-of-domain questions. These are candidates and still require review.
OOD_QUESTIONS = [
    "How do I sort a list of dictionaries by a specific key in Python?",
    "Why does my Python program return None unexpectedly?",
    "What is the difference between a stack and a queue?",
    "How can I explain binary search with a simple example?",
    "What causes a memory leak in a software application?",
    "How does a database index improve query speed?",
    "What is the difference between supervised and unsupervised learning?",
    "How can I detect overfitting in a machine learning model?",
    "What does a confusion matrix measure in a classification task?",
    "How do convolutional neural networks process images?",
    "Why does the sky appear blue during the day?",
    "What is the difference between weather and climate?",
    "How does the water cycle work?",
    "Why do earthquakes occur near tectonic plate boundaries?",
    "What is the role of chlorophyll in plants?",
    "How do solar panels convert sunlight into electricity?",
    "What causes ocean tides?",
    "Why does metal expand when heated?",
    "How do vaccines help the immune system?",
    "What is the difference between a virus and a bacterium?",
    "How do I calculate compound interest on a monthly deposit?",
    "What is the difference between mean, median, and mode?",
    "How do I calculate the probability of rolling a six on a die?",
    "Can you explain standard deviation using a simple dataset?",
    "How do I convert a decimal number into binary?",
    "What is the difference between correlation and causation?",
    "How can I solve a quadratic equation?",
    "What is the Pythagorean theorem used for?",
    "How do I calculate the area of a circle?",
    "What does a logarithm represent?",
    "What should I pack for a three-day trip to the mountains?",
    "How can I plan a low-cost weekend trip?",
    "What is the best way to avoid jet lag on a long flight?",
    "How do I read a railway timetable?",
    "What should I consider when choosing a travel backpack?",
    "How can I make a simple vegetarian meal with lentils?",
    "Why does bread dough need time to rise?",
    "How should fresh herbs be stored to stay usable longer?",
    "What is the difference between baking powder and baking soda?",
    "How can I reduce food waste at home?",
    "How do I create a weekly study timetable?",
    "What techniques can help me remember technical concepts?",
    "How can I practise speaking English more confidently?",
    "How should I structure a presentation for a college project?",
    "What makes a good project README file?",
    "How can I improve battery life on a laptop?",
    "Why does a Wi-Fi connection become unstable?",
    "What is the difference between RAM and storage?",
    "How do I back up important files to an external drive?",
    "Why is my laptop fan making more noise than usual?",
    "How do I start a small vegetable garden on a balcony?",
    "Which conditions help tomato plants grow well?",
    "How often should indoor plants generally be watered?",
    "What is compost and how is it made?",
    "How can rainwater be collected for gardening?",
    "What are the main differences between renewable and non-renewable energy?",
    "How does a wind turbine generate electricity?",
    "What is the greenhouse effect?",
    "How can cities reduce traffic congestion?",
    "Why is biodiversity important for an ecosystem?",
    "How do I make a simple monthly household budget?",
    "What factors should I compare when choosing a savings account?",
    "What is the difference between a debit card and a credit card?",
    "How does inflation affect the purchasing power of money?",
    "What does diversification mean in investing?",
    "How do I create a bar chart in Excel?",
    "How can I remove duplicate rows in a spreadsheet?",
    "What is the difference between a CSV file and a JSON file?",
    "How do I use Git branches when collaborating on code?",
    "What is an API and how does a client use it?",
    "How do I prepare for a technical interview?",
    "What should a beginner include in a software portfolio?",
    "How can a team divide tasks during a hackathon?",
    "What is the difference between a compiler and an interpreter?",
    "How can I write clearer comments in code?",
]

def main():
    existing = get_existing_questions()
    print("Existing questions excluded from exact-match reuse:", len(existing))

    passages = get_passages()
    print("Unique legal passages available:", len(passages))
    print("Passages by source:", {
        label: sum(p["source"] == label for p in passages)
        for label in ["constitution", "consumer_protection"]
    })

    candidate_rows = []
    seen_questions = set(existing)

    # Generate up to 170 candidates per legal domain.
    for label, target_passages in [
        ("constitution", 95),
        ("consumer_protection", 100),
    ]:
        source_passages = [p for p in passages if p["source"] == label]
        if len(source_passages) < target_passages:
            target_passages = len(source_passages)

        selected = random.sample(source_passages, target_passages)
        passage_items = []
        for i, p in enumerate(selected):
            passage_items.append({
                "id": f"{label[:3].upper()}{i + 1:03d}",
                **p,
            })

        generated = []
        for start in range(0, len(passage_items), 5):
            batch = passage_items[start:start + 5]
            try:
                result = ollama_questions(batch, label)
                for item in result.get("items", []):
                    questions = item.get("questions", [])
                    for q in questions:
                        q = str(q).strip()
                        if q and q.endswith("?"):
                            generated.append((item.get("id"), q))
                print(f"{label}: processed {min(start + 5, len(passage_items))}/{len(passage_items)} passages")
            except Exception as exc:
                print(f"Warning: generation batch failed for {label}, batch {start}: {exc}")

            time.sleep(0.15)

        passage_lookup = {p["id"]: p for p in passage_items}
        accepted = 0

        for passage_id, question in generated:
            key = normalize(question)
            if not key or key in seen_questions:
                continue
            if len(question.split()) < 5:
                continue

            passage = passage_lookup.get(passage_id)
            if not passage:
                continue

            seen_questions.add(key)
            candidate_rows.append({
                "text": question,
                "label": label,
                "source": label,
                "page": passage["page"],
                "chunk_id": passage["chunk_id"],
                "supporting_passage": passage["context"],
                "review_status": "pending",
                "candidate_origin": "ollama_generated_from_legal_passage",
            })
            accepted += 1
            if accepted >= 170:
                break

        print(f"{label}: accepted {accepted}/170 candidates after exact-match filtering")

    # Add varied OOD candidates, excluding exact overlaps.
    ood_accepted = 0
    random.shuffle(OOD_QUESTIONS)
    for question in OOD_QUESTIONS:
        key = normalize(question)
        if not key or key in seen_questions:
            continue
        seen_questions.add(key)
        candidate_rows.append({
            "text": question,
            "label": "out_of_domain",
            "source": "out_of_domain",
            "page": None,
            "chunk_id": None,
            "supporting_passage": "",
            "review_status": "pending",
            "candidate_origin": "curated_general_topic_candidate",
        })
        ood_accepted += 1
        if ood_accepted >= 60:
            break

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in candidate_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== CANDIDATE FILE CREATED ===")
    print(OUTPUT_FILE)
    print("Total candidates:", len(candidate_rows))
    print("Counts:", {
        label: sum(row["label"] == label for row in candidate_rows)
        for label in ["constitution", "consumer_protection", "out_of_domain"]
    })
    print("\nAll generated candidates remain pending review.")
    print("Do not use this file as a final test set until reviewed.")

if __name__ == "__main__":
    main()
