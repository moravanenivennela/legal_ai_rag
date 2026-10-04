import os
import json
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_DIR = Path("fine_tuning/dataset")
OUTPUT_DIR = Path("models/legal_lora")

TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"

MAX_LENGTH = 512

# CPU-safe settings
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
EPOCHS = 1
LEARNING_RATE = 2e-4


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("LEGAL LLM LoRA FINE-TUNING")
print("=" * 60)

print(f"Base model : {BASE_MODEL}")
print(f"Device     : {DEVICE}")
print(f"CUDA       : {torch.cuda.is_available()}")
print(f"Train file : {TRAIN_FILE}")
print(f"Val file   : {VALIDATION_FILE}")
print("=" * 60)


# ============================================================
# CHECK DATASET
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training dataset not found: {TRAIN_FILE}")

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation dataset not found: {VALIDATION_FILE}"
    )


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


train_records = load_jsonl(TRAIN_FILE)
validation_records = load_jsonl(VALIDATION_FILE)

print(f"Training examples   : {len(train_records)}")
print(f"Validation examples : {len(validation_records)}")

if len(train_records) == 0:
    raise ValueError("Training dataset is empty.")


# ============================================================
# CONVERT DATA TO INSTRUCTION FORMAT
# ============================================================

def build_prompt(record):

    instruction = record.get(
        "instruction",
        "Answer the legal question using only the provided legal context."
    )

    input_text = record.get("input", "")
    output_text = record.get("output", "")

    text = (
        "<|im_start|>system\n"
        "You are a legal question answering assistant. "
        "Answer using only the supplied legal context.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n\n"
        f"{input_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output_text}\n"
        "<|im_end|>"
    )

    return text


train_texts = [
    build_prompt(record)
    for record in train_records
]

validation_texts = [
    build_prompt(record)
    for record in validation_records
]


train_dataset = Dataset.from_dict({
    "text": train_texts
})

validation_dataset = Dataset.from_dict({
    "text": validation_texts
})


print("\nDataset conversion complete.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading ${BASE_MODEL}...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.to(DEVICE)
model.config.use_cache = False


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\nApplying LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    bias="none",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


print("\nTokenizing training dataset...")

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print("Tokenizing validation dataset...")

tokenized_validation = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=str(OUTPUT_DIR),

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    logging_steps=1,

    save_strategy="epoch",

    eval_strategy="epoch",

    save_total_limit=2,

    report_to="none",

    fp16=False,

    bf16=False,

    dataloader_num_workers=0,

    gradient_checkpointing=True,

    remove_unused_columns=False,

)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    processing_class=tokenizer,

    data_collator=data_collator,
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING LoRA TRAINING")
print("=" * 60)

print("This is running on CPU.")
print("The first run may take a long time.")

train_result = trainer.train()


# ============================================================
# SAVE ADAPTER
# ============================================================

print("\nSaving LoRA adapter...")

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)


# ============================================================
# SAVE TRAINING METRICS
# ============================================================

metrics = train_result.metrics

with open(
    OUTPUT_DIR / "training_metrics.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
        default=str,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("LoRA TRAINING COMPLETE")
print("=" * 60)

print(f"Adapter saved to:")
print(OUTPUT_DIR)

print("\nTraining metrics:")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nNext stage:")
print("1. Evaluate the LoRA adapter")
print("2. Compare base vs fine-tuned model")
print("3. Merge adapter if required")
print("4. Import the resulting model into Ollama")
print("5. Connect it to the existing Hybrid RAG pipeline")
print("=" * 60)
