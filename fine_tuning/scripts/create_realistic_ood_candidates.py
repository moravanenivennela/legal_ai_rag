import json
import random
import re
from pathlib import Path

SEED = 20261002
OUT = Path("fine_tuning/dataset/expanded/classification")
OUTPUT = OUT / "realistic_ood_candidates.jsonl"

# Varied, realistic non-legal requests. These are synthetic candidates
# and must be reviewed before being used for training or evaluation.
QUESTIONS = [
    # Programming and software
    "Why does my Python program raise an IndexError?",
    "Help me understand the difference between a list and a tuple in Python.",
    "How can I debug a React component that fails to render?",
    "Explain how Git branches work when collaborating on a project.",
    "What causes a database query to become slow?",
    "How do I deploy a small web application?",
    "What is the purpose of a virtual environment in Python?",
    "How can I reduce memory usage in a machine learning model?",
    "Explain the difference between supervised and unsupervised learning.",
    "How do I fix a Java program that cannot find its main method?",
    "What should I include in a software project README?",
    "How can I design a responsive navigation menu?",
    "Why might an API request return a 404 error?",
    "How do unit tests help prevent software bugs?",
    "Explain the difference between CPU and GPU computation.",

    # Everyday tasks and writing
    "Help me prepare a weekly grocery list for two people.",
    "Suggest a simple vegetarian dinner using rice and vegetables.",
    "How can I organize my study schedule before exams?",
    "Rewrite this paragraph to sound more professional.",
    "Give me ideas for a low-cost birthday celebration.",
    "How can I remove a coffee stain from a cotton shirt?",
    "Suggest a packing checklist for a three-day trip.",
    "What are some ways to improve my public speaking?",
    "Help me create a monthly personal budget.",
    "How can I compare two laptop specifications?",
    "Suggest activities for a rainy weekend at home.",
    "How should I prepare for a job interview?",
    "Give me a checklist for moving into a new apartment.",
    "How can I plan a beginner-friendly home workout?",
    "Suggest ways to reduce food waste in my kitchen.",

    # Science, nature, and education
    "Why does the Moon appear to change shape during the month?",
    "Explain how photosynthesis works in simple terms.",
    "What causes ocean tides?",
    "How do solar panels convert sunlight into electricity?",
    "Why do some materials conduct electricity better than others?",
    "Explain the difference between weather and climate.",
    "How do birds navigate during seasonal migration?",
    "What is the role of bacteria in food fermentation?",
    "How does a telescope collect light from distant objects?",
    "Why do earthquakes occur near tectonic plate boundaries?",
    "Explain how sound waves travel through air.",
    "What is the difference between mass and weight?",
    "How do plants respond to changes in daylight?",
    "Explain the basic idea behind probability distributions.",
    "What are the stages of the water cycle?",

    # Travel, arts, and recreation
    "Help me plan a two-day visit to a hill station.",
    "What should a beginner learn before playing badminton?",
    "How can I practice taking better photographs with a phone?",
    "Suggest a reading plan for someone returning to novels.",
    "What equipment is useful for a beginner gardener?",
    "How can I learn basic conversational Spanish?",
    "Suggest indoor games for a small group of friends.",
    "What should I consider when choosing a bicycle?",
    "How can I care for indoor plants during hot weather?",
    "Explain the basic rules of chess to a beginner.",
    "What are some ways to improve drawing proportions?",
    "How can I build a simple daily meditation habit?",
    "Suggest a beginner itinerary for exploring a new city.",
    "What should I consider when selecting a musical instrument?",
    "How can I start learning to bake bread at home?",

    # General information and practical questions
    "What factors should I compare when choosing a mobile phone?",
    "How does compound interest work in a savings account?",
    "Explain how electric vehicles store and use energy.",
    "What are the differences between renewable energy sources?",
    "How can I recognize misleading statistics in a chart?",
    "What is the purpose of a password manager?",
    "How can I protect important files from accidental deletion?",
    "Explain how recycling facilities sort household waste.",
    "What should I look for when comparing internet plans?",
    "How can I make a presentation easier for an audience to follow?",
    "What are common causes of a home Wi-Fi connection dropping?",
    "How do noise-cancelling headphones work?",
    "What is the difference between a debit card and a credit card?",
    "How can I create a simple inventory spreadsheet?",
    "Explain how public transportation route maps are organized.",

    # Queries that deliberately resemble legal language but are not
    # questions answerable from the project's two supported legal sources.
    "Explain the basics of copyright protection for a photograph.",
    "How do international trade agreements affect import costs?",
    "What are the main features of a country's monetary policy?",
    "Explain how workplace safety inspections generally operate.",
    "How does a university typically calculate a semester grade?",
    "What is the difference between a patent and a trade secret?",
    "How do building permits generally work for home renovations?",
    "Explain how international organizations coordinate disaster relief.",
    "What factors affect the price of agricultural commodities?",
    "How do insurance companies estimate vehicle repair costs?",
]

def norm(value):
    return re.sub(r"\s+", " ", value.lower()).strip()

def question_from(row):
    return str(
        row.get("text")
        or row.get("question")
        or row.get("input")
        or ""
    ).strip()

existing = set()
source_files = [
    OUT / "train_candidates.jsonl",
    OUT / "validation_candidates.jsonl",
    OUT / "test_candidates.jsonl",
    OUT / "repaired" / "train_classification.jsonl",
    OUT / "repaired" / "validation_classification.jsonl",
    OUT / "repaired" / "test_classification.jsonl",
    OUT / "repaired_v2" / "train_classification.jsonl",
    OUT / "repaired_v2" / "validation_classification.jsonl",
    OUT / "repaired_v2" / "test_classification.jsonl",
]

for path in source_files:
    if not path.exists():
        continue
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                q = question_from(row)
                if q:
                    existing.add(norm(q))
            except (json.JSONDecodeError, TypeError):
                pass

seen = set()
records = []
for question in QUESTIONS:
    key = norm(question)
    if key in existing or key in seen:
        continue
    seen.add(key)
    records.append({
        "text": question,
        "label": "out_of_domain",
        "source": "synthetic_realistic_ood_candidate",
        "page": "",
        "split": "candidate_only",
        "review_status": "pending",
    })

random.Random(SEED).shuffle(records)
OUT.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("REALISTIC OOD CANDIDATE GENERATION")
print(f"Questions drafted: {len(QUESTIONS)}")
print(f"Unique new candidates saved: {len(records)}")
print(f"Output: {OUTPUT}")
print("Existing dataset files were not modified.")
print("All new candidates require review before use.")
