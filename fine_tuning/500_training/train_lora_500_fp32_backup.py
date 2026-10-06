from pathlib import Path
import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "fine_tuning" / "dataset_500_20261004_214536"
OUTPUT_DIR = ROOT / "models" / "legal_lora_500"

# Qwen 2.5 3B is the same model family already used in the
# legal-QA generation workflow.
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def format_example(row):
    question = row.get("question", "").strip()
    answer = row.get("answer", "").strip()

    return (
        "### Legal Question:\n"
        f"{question}\n\n"
        "### Legal Answer:\n"
        f"{answer}"
    )


print("=" * 70)
print("LEGAL AI RAG — 500 QA LoRA FINE-TUNING")
print("=" * 70)

train_rows = load_jsonl(DATA_DIR / "train.jsonl")
val_rows = load_jsonl(DATA_DIR / "validation.jsonl")

print(f"Training examples   : {len(train_rows)}")
print(f"Validation examples : {len(val_rows)}")
print(f"CUDA available      : {torch.cuda.is_available()}")

if len(train_rows) != 400:
    raise RuntimeError("Training dataset must contain exactly 400 records.")

if len(val_rows) != 50:
    raise RuntimeError("Validation dataset must contain exactly 50 records.")

train_texts = [{"text": format_example(x)} for x in train_rows]
val_texts = [{"text": format_example(x)} for x in val_rows]

train_dataset = Dataset.from_list(train_texts)
val_dataset = Dataset.from_list(val_texts)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

training_args = TrainingArguments(
    output_dir=str(ROOT / "fine_tuning" / "500_training" / "checkpoints"),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,
    weight_decay=0.01,

    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    save_steps=25,
    save_total_limit=2,

    report_to="none",

    fp16=False,
    bf16=False,

    dataloader_num_workers=0,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
)

print("\nStarting LoRA training...")
print("This is CPU training; it may take significant time.")
print("=" * 70)

trainer.train()

print("\nSaving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTRAINING COMPLETE")
print(f"Adapter saved to: {OUTPUT_DIR}")
