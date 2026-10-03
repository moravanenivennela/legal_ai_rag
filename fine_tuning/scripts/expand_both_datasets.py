import json
import random
import re
import time
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "fine_tuning" / "dataset"
OUT = DATA / "expanded"
MODEL = "qwen2.5:3b"
SEED = 42

# Initial run: adjust later after checking quality and runtime.
MAX_TRAIN_PER_SOURCE = {
    "constitution": 200,
    "consumer_protection": 75,
}
MAX_VALIDATION_PER_SOURCE = 15
MAX_CLASSIFICATION_TEST_PER_SOURCE = 15

for folder in ["classification", "legal_qa"]:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def read_jsonl(path):
    records = []
    if not path.exists():
        print(f"Missing file: {path}")
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Skipping invalid JSON:", path)
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_passages(filename):
    return read_jsonl(DATA / "passage_backup" / filename)


def clean_question(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def call_generator(passage):
    source = passage["source"]
    context = passage["context"][:4500]

    prompt = f"""
You are creating candidate training data from an Indian legal source.

SOURCE LABEL: {source}
PASSAGE:
{context}

Return only a valid JSON object with this schema:
{{
  "classification_questions": [
    "question 1",
    "question 2"
  ],
  "qa_pairs": [
    {{
      "question": "one answerable question",
      "answer": "a concise answer supported by the passage",
      "legal_reference": "reference only if explicitly present in passage"
    }},
    {{
      "question": "a different answerable question",
      "answer": "a concise answer supported by the passage",
      "legal_reference": "reference only if explicitly present in passage"
    }}
  ]
}}

Rules:
- Questions must be specific and answerable from the passage.
- Do not invent section numbers, article numbers, dates, penalties, or legal rules.
- Answers must use only the passage.
- If no reliable answer can be extracted, return empty arrays.
- Do not include markdown or text outside JSON.
"""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Generate cautious legal dataset candidates. Never invent legal facts."
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0.1},
    )
    return json.loads(response["message"]["content"])


def passage_key(row):
    return (
        row.get("source"),
        str(row.get("page")),
        str(row.get("chunk_id")),
    )


def generate_split(passages, split, classification_rows, qa_rows,
                   limit_by_source, skip_keys=None):
    skip_keys = skip_keys or set()
    counts = {"constitution": 0, "consumer_protection": 0}

    # Use deterministic ordering and stable passage-level splits.
    grouped = {
        "constitution": [],
        "consumer_protection": [],
    }
    for passage in passages:
        source = passage.get("source")
        if source in grouped and passage_key(passage) not in skip_keys:
            grouped[source].append(passage)

    for source in grouped:
        random.Random(f"{SEED}-{split}-{source}").shuffle(grouped[source])

    seen_questions = set()
    # Prevent duplicate candidate records within this run.
    for source in grouped:
        cap = limit_by_source.get(source, 0)
        selected = grouped[source][:cap]

        for passage in selected:
            key = passage_key(passage)
            print(f"[{split}] {source}: passage {counts[source] + 1}/{len(selected)}")

            try:
                generated = call_generator(passage)
            except Exception as exc:
                print("Generation failed; skipping passage:", exc)
                continue

            page = passage.get("page", "")
            context = passage.get("context", "")

            for question in generated.get("classification_questions", []):
                question = clean_question(question)
                qkey = question.casefold()
                if len(question) < 12 or qkey in seen_questions:
                    continue
                seen_questions.add(qkey)
                classification_rows.append({
                    "text": question,
                    "label": source,
                    "source": source,
                    "page": page,
                    "split": split,
                    "review_status": "pending",
                })

            for item in generated.get("qa_pairs", []):
                question = clean_question(item.get("question"))
                answer = re.sub(
                    r"\s+", " ", str(item.get("answer", "") or "")
                ).strip()
                reference = str(item.get("legal_reference", "") or "").strip()

                if len(question) < 12 or len(answer) < 5:
                    continue

                qa_rows.append({
                    "instruction": (
                        "Answer the legal question using only the supplied "
                        "legal evidence. Do not invent legal provisions or "
                        "facts. If evidence is insufficient, say so."
                    ),
                    "input": f"Question: {question}\n\nLegal evidence:\n{context}",
                    "output": answer,
                    "evidence": context,
                    "source": source,
                    "page": str(page),
                    "legal_reference": reference,
                    "review_status": "pending",
                    "split": split,
                })

            counts[source] += 1

    return counts


def main():
    train_passages = load_passages("train_passages.jsonl")
    validation_passages = load_passages("validation_passages.jsonl")
    test_passages = load_passages("test_passages.jsonl")

    print("Model:", MODEL)
    print("Train passages:", len(train_passages))
    print("Validation passages:", len(validation_passages))
    print("Test passages:", len(test_passages))
    print("Existing datasets will not be overwritten.")

    classification = {"train": [], "validation": [], "test": []}
    qa = {"train": [], "validation": []}

    train_counts = generate_split(
        train_passages,
        "train",
        classification["train"],
        qa["train"],
        MAX_TRAIN_PER_SOURCE,
    )

    validation_counts = generate_split(
        validation_passages,
        "validation",
        classification["validation"],
        qa["validation"],
        {
            "constitution": MAX_VALIDATION_PER_SOURCE,
            "consumer_protection": MAX_VALIDATION_PER_SOURCE,
        },
    )

    test_counts = generate_split(
        test_passages,
        "test",
        classification["test"],
        [],  # No QA generation from the held-out test passages.
        {
            "constitution": MAX_CLASSIFICATION_TEST_PER_SOURCE,
            "consumer_protection": MAX_CLASSIFICATION_TEST_PER_SOURCE,
        },
    )

    # Add deterministic out-of-domain questions. These are synthetic
    # candidates, not user-collected examples, and need review too.
    ood_topics = [
        "astronomy", "music", "cricket", "cooking", "geography",
        "mathematics", "programming", "gardening", "physics",
        "cinema", "travel", "history", "biology", "weather",
        "photography", "sports", "literature", "chemistry",
        "computer hardware", "space exploration", "language learning",
        "renewable energy", "ocean science", "animal behaviour",
        "architecture", "gaming", "statistics", "robotics",
        "nutrition", "transportation",
    ]
    ood_templates = [
        "Explain the basic principles of {}.",
        "How does {} work?",
        "What are common applications of {}?",
        "Give a beginner overview of {}.",
        "What are the main concepts in {}?",
    ]

    for split in ["train", "validation", "test"]:
        topics = ood_topics.copy()
        random.Random(f"{SEED}-ood-{split}").shuffle(topics)
        # Separate questions by split, avoiding shared question strings.
        for topic in topics:
            for template in ood_templates:
                question = template.format(topic)
                classification[split].append({
                    "text": question,
                    "label": "out_of_domain",
                    "source": "synthetic_ood",
                    "page": "",
                    "split": split,
                    "review_status": "pending",
                })

    class_dir = OUT / "classification"
    qa_dir = OUT / "legal_qa"

    for split, rows in classification.items():
        write_jsonl(class_dir / f"{split}_candidates.jsonl", rows)

    for split, rows in qa.items():
        write_jsonl(qa_dir / f"{split}_candidates.jsonl", rows)

    summary = {
        "model": MODEL,
        "train_passage_counts_processed": train_counts,
        "validation_passage_counts_processed": validation_counts,
        "test_passage_counts_processed": test_counts,
        "classification_counts": {
            split: {
                label: sum(r["label"] == label for r in rows)
                for label in [
                    "constitution", "consumer_protection", "out_of_domain"
                ]
            }
            for split, rows in classification.items()
        },
        "qa_counts": {split: len(rows) for split, rows in qa.items()},
        "note": (
            "All generated records are candidates and require legal review. "
            "Test classification candidates must remain out of training. "
            "QA test data was not generated or changed."
        ),
    }

    (OUT / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n===== GENERATION SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nCandidates saved under:", OUT)


if __name__ == "__main__":
    main()
