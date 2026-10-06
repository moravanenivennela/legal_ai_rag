import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(".")
TEST = ROOT / "fine_tuning/dataset_500_20261004_214536/test.jsonl"
ADAPTER = ROOT / "models/legal_lora_500"
OUT = ROOT / "reports/legal_qa_500_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def scores(pred, ref):
    p = normalize(pred).split()
    r = normalize(ref).split()

    if not p or not r:
        return 0.0, 0.0, 0.0

    pc = Counter(p)
    rc = Counter(r)
    overlap = sum((pc & rc).values())

    precision = overlap / len(p)
    recall = overlap / len(r)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return precision, recall, f1

print("=" * 70)
print("LEGAL AI RAG — REAL 50-QUESTION TEST EVALUATION")
print("=" * 70)

with open(TEST, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]

print("Test questions:", len(rows))
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading 4-bit base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

device = next(model.parameters()).device
print("Model device:", device)

results = []

for i, row in enumerate(rows, 1):
    question = row["input"]
    reference = row["output"]
    evidence = row.get("evidence", "")

    prompt = (
        "Answer the legal question using only the provided legal context.\n\n"
        + question
        + "\n\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    elapsed = time.time() - start

    precision, recall, f1 = scores(prediction, reference)

    exact = int(normalize(prediction) == normalize(reference))

    evidence_norm = normalize(evidence)
    prediction_norm = normalize(prediction)

    evidence_support = int(
        bool(evidence_norm) and
        any(
            len(part) >= 40 and part in evidence_norm
            for part in re.split(r"[.!?]", prediction_norm)
        )
    )

    results.append({
        "id": i,
        "question": question,
        "reference_answer": reference,
        "prediction": prediction,
        "source": row.get("source"),
        "page": row.get("page"),
        "chunk_id": row.get("chunk_id"),
        "question_type": row.get("question_type"),
        "exact_match": exact,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evidence_support": evidence_support,
        "generation_seconds": elapsed,
    })

    print(
        f"[{i:02d}/{len(rows)}] "
        f"F1={f1:.4f} "
        f"EM={exact} "
        f"time={elapsed:.1f}s"
    )

    with open(
        OUT / "test_predictions.jsonl",
        "a",
        encoding="utf-8"
    ) as f:
        f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")

# -----------------------------
# Aggregate metrics
# -----------------------------

n = len(results)

metrics = {
    "test_examples": n,
    "exact_match": sum(x["exact_match"] for x in results) / n,
    "precision": sum(x["precision"] for x in results) / n,
    "recall": sum(x["recall"] for x in results) / n,
    "f1": sum(x["f1"] for x in results) / n,
    "evidence_support": sum(x["evidence_support"] for x in results) / n,
    "avg_generation_seconds": sum(x["generation_seconds"] for x in results) / n,
}

with open(
    OUT / "fine_tuned_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=2)

with open(
    OUT / "fine_tuned_metrics.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in metrics.items():
        writer.writerow([k, v])

print("\n" + "=" * 70)
print("FINE-TUNED MODEL RESULTS")
print("=" * 70)

for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k:30s}: {v:.4f}")
    else:
        print(f"{k:30s}: {v}")

print("\nSaved:")
print(OUT / "test_predictions.jsonl")
print(OUT / "fine_tuned_metrics.json")
print(OUT / "fine_tuned_metrics.csv")
