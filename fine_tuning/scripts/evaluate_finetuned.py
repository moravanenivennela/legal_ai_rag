
import json
import re
import string
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path("models/legal_lora_large_20261002")
TEST_FILE = Path("fine_tuning/dataset/qa_test.jsonl")

OUTPUT_FILE = Path("fine_tuning/evaluation_large_adapter.json")
GRAPH_FILE = Path("outputs/09_large_adapter_test_metrics.png")
DETAILS_FILE = Path("outputs/09_large_adapter_test_details.json")

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 160

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def normalize(text):
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(expected, generated):
    """Whitespace-token F1; a text-overlap metric, not legal correctness."""
    expected_tokens = normalize(expected).split()
    generated_tokens = normalize(generated).split()

    if not expected_tokens or not generated_tokens:
        return 0.0

    from collections import Counter

    expected_counts = Counter(expected_tokens)
    generated_counts = Counter(generated_tokens)
    overlap = sum((expected_counts & generated_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


def load_test_record(record):
    # Supports both the new question/answer format and older input/output format.
    question = record.get("question") or record.get("input") or ""
    expected = record.get("answer") or record.get("output") or ""

    context = (
        record.get("supporting_passage")
        or record.get("context")
        or record.get("evidence")
        or ""
    )

    # If an older record stores the complete prompt in "input", use it as a fallback.
    if not question and record.get("input"):
        question = record["input"]

    return str(question).strip(), str(context).strip(), str(expected).strip()


# --------------------------------------------------
# VALIDATE FILES
# --------------------------------------------------
print("=" * 65)
print("APPROVED LEGAL LLM - HELD-OUT TEST EVALUATION")
print("=" * 65)
print(f"Base model : {BASE_MODEL}")
print(f"Adapter    : {ADAPTER_DIR}")
print(f"Test file  : {TEST_FILE}")

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"New adapter not found: {ADAPTER_DIR}. "
        "Check that training completed successfully."
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found: {TEST_FILE}. "
        "Check the filename before running evaluation."
    )

with TEST_FILE.open("r", encoding="utf-8") as f:
    tests = [json.loads(line) for line in f if line.strip()]

if not tests:
    raise ValueError("The test dataset is empty.")

prepared_tests = [load_test_record(record) for record in tests]

for index, (question, context, expected) in enumerate(prepared_tests, 1):
    if not question or not expected:
        raise ValueError(
            f"Test record {index} is missing a question or reference answer. "
            "Inspect the dataset schema before evaluating."
        )

print(f"Test examples: {len(prepared_tests)}")
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("Loading the newly trained LoRA adapter...")
model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
model.eval()
model.to("cpu")

# --------------------------------------------------
# GENERATE ANSWERS
# --------------------------------------------------
results = []
start_time = time.time()

for i, (question, context, expected) in enumerate(prepared_tests, 1):
    prompt_context = context if context else "No supporting passage was provided."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal question answering assistant. "
                "Answer using only the supplied legal context. "
                "If the context does not contain the answer, say so. "
                "Do not invent legal provisions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Legal context:\n{prompt_context}\n\n"
                f"Question:\n{question}"
            ),
        },
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = (
            f"System: {messages[0]['content']}\n"
            f"User: {messages[1]['content']}\n"
            "Assistant:"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    exact_match = int(normalize(expected) == normalize(generated))
    f1 = token_f1(expected, generated)

    result = {
        "index": i,
        "source": tests[i - 1].get("source", ""),
        "page": tests[i - 1].get("page", ""),
        "question": question,
        "supporting_passage": context,
        "expected": expected,
        "generated": generated,
        "exact_match": exact_match,
        "token_f1": round(f1, 4)
    }
    results.append(result)

    print(f"\n[{i}/{len(prepared_tests)}] Token F1: {f1:.3f}")
    print("Question:", question)
    print("Expected:", expected)
    print("Generated:", generated)

    # Save progress after every example, so partial results survive interruption.
    partial = {
        "base_model": BASE_MODEL,
        "adapter": str(ADAPTER_DIR),
        "test_file": str(TEST_FILE),
        "completed_examples": len(results),
        "results": results
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(partial, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# AGGREGATE METRICS
# --------------------------------------------------
count = len(results)
exact_match_rate = sum(r["exact_match"] for r in results) / count
average_f1 = sum(r["token_f1"] for r in results) / count
elapsed_seconds = time.time() - start_time

summary = {
    "base_model": BASE_MODEL,
    "adapter": str(ADAPTER_DIR),
    "test_file": str(TEST_FILE),
    "test_examples": count,
    "exact_match_rate": round(exact_match_rate, 4),
    "average_token_f1": round(average_f1, 4),
    "evaluation_runtime_seconds": round(elapsed_seconds, 2),
    "metric_note": (
        "Exact match and token F1 measure text overlap. "
        "They do not establish legal correctness or factual validity."
    ),
    "results": results
}

OUTPUT_FILE.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

DETAILS_FILE.parent.mkdir(parents=True, exist_ok=True)
DETAILS_FILE.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# --------------------------------------------------
# TEST METRICS GRAPH
# --------------------------------------------------
GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)

metric_names = ["Exact match rate", "Average token F1"]
metric_values = [exact_match_rate * 100, average_f1 * 100]

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_names, metric_values)
plt.ylim(0, 100)
plt.ylabel("Score (%)")
plt.title(f"Held-out Test Evaluation (n={count})")
plt.grid(axis="y", linestyle="--", alpha=0.35)

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.figtext(
    0.5, 0.01,
    "Text-overlap metrics only; not a measure of legal correctness.",
    ha="center",
    fontsize=8
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(GRAPH_FILE, dpi=300, bbox_inches="tight")
plt.close()

print("\n" + "=" * 65)
print("TEST EVALUATION COMPLETE")
print("=" * 65)
print(f"Test examples       : {count}")
print(f"Exact match rate    : {exact_match_rate * 100:.2f}%")
print(f"Average token F1    : {average_f1 * 100:.2f}%")
print(f"Evaluation time     : {elapsed_seconds / 60:.2f} minutes")
print(f"Metrics JSON        : {OUTPUT_FILE}")
print(f"Per-example results : {DETAILS_FILE}")
print(f"Test metrics graph  : {GRAPH_FILE}")