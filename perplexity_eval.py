import torch
import math
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model_id, text_samples):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    total_loss = 0.0
    total_count = 0
    
    for text in text_samples:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_count += inputs["input_ids"].size(1)
            
    return math.exp(total_loss / total_count) if total_count > 0 else float('inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="Number of test samples")
    args = parser.parse_args()

    sample_legal_texts = [
        "The Supreme Court shall be a court of record and shall have all the powers of such a court.",
        "Every consumer has the right to be protected against unfair trade practices under the Act.",
        "When a dispute is referred to mediation, the mediator shall assist the parties in reaching an amicable settlement."
    ][:args.limit]

    print(f"Loaded {len(sample_legal_texts)} sample texts for perplexity evaluation.")
