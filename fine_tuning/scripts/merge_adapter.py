import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path("models/legal_lora")
MERGED_DIR = Path("models/merged")

print("=" * 60)
print("MERGING QWEN + LEGAL LoRA ADAPTER")
print("=" * 60)

print(f"Base model : {BASE_MODEL}")
print(f"Adapter    : {ADAPTER_DIR}")
print(f"Output     : {MERGED_DIR}")

if not ADAPTER_DIR.exists():
    raise FileNotFoundError("LoRA adapter not found.")

MERGED_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_DIR)
)

print("Merging adapter into base model...")

model = model.merge_and_unload()

print("Saving merged model...")

model.save_pretrained(
    MERGED_DIR,
    safe_serialization=True
)

tokenizer.save_pretrained(MERGED_DIR)

print("\n" + "=" * 60)
print("MERGE COMPLETE")
print("=" * 60)
print(f"Merged model saved to: {MERGED_DIR}")
