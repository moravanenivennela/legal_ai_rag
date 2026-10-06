import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = "fine_tuning/dataset_500_20261004_214536/train.jsonl"
VAL_FILE = "fine_tuning/dataset_500_20261004_214536/validation.jsonl"

OUTPUT_DIR = "models/legal_lora_500_answer_only"
CHECKPOINT_DIR = "fine_tuning/500_training/checkpoints_answer_only"

MAX_LENGTH = 768

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("LEGAL AI - ANSWER-ONLY QLoRA FINE-TUNING")
print("=" * 70)

print("\nLoading datasets...")

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
)["train"]

val_dataset = load_dataset(
    "json",
    data_files=VAL_FILE,
)["train"]

print(f"Train examples      : {len(train_dataset)}")
print(f"Validation examples : {len(val_dataset)}")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

print("\nPreparing answer-only dataset...")

def tokenize_example(example):
    instruction = str(example.get("instruction", "")).strip()
    user_input = str(example.get("input", "")).strip()
    answer = str(example.get("output", "")).strip()

    prompt = (
        instruction
        + "\n\n"
        + user_input
        + "\n\nAnswer:"
    )

    full_text = prompt + " " + answer

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_length = min(
        len(prompt_tokens["input_ids"]),
        len(input_ids),
    )

    labels = (
        [-100] * prompt_length
        + input_ids[prompt_length:]
    )

    # Make sure at least one answer token is trainable.
    if all(x == -100 for x in labels):
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = train_dataset.map(
    tokenize_example,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training data",
)

val_dataset = val_dataset.map(
    tokenize_example,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation data",
)

print("Tokenization complete.")

print("\nLoading Qwen2.5-3B in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": "cpu"},
    trust_remote_code=True,
)

model.config.use_cache = False

print("Model loaded.")

print("\nPreparing model for QLoRA...")

model = prepare_model_for_kbit_training(model)

model.enable_input_require_grads()

model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()

class LegalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(
            len(feature["input_ids"])
            for feature in features
        )

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])

            input_ids.append(
                feature["input_ids"]
                + [self.tokenizer.pad_token_id] * pad_len
            )

            attention_masks.append(
                feature["attention_mask"]
                + [0] * pad_len
            )

            labels.append(
                feature["labels"]
                + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


data_collator = LegalDataCollator(tokenizer)

print("\nStarting training...")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,
    weight_decay=0.01,


    logging_strategy="steps",
    logging_steps=5,

    eval_strategy="steps",
    eval_steps=25,

    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",

    fp16=False,
    bf16=False,

    dataloader_pin_memory=False,

    remove_unused_columns=False,

    gradient_checkpointing=True,

    optim="adamw_torch",

    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

print("\nSaving BEST answer-only LoRA adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Adapter saved to: {OUTPUT_DIR}")
print("Best checkpoint selected using validation loss.")
print("=" * 70)
