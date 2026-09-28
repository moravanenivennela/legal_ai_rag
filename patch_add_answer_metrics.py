with open("rag_engine.py", "r", encoding="utf-8") as f:
    content = f.read()

old_anchor = "        return round(max(0.0, min(100.0, avg_entailment * 100)), 1)"

new_block = '''        return round(max(0.0, min(100.0, avg_entailment * 100)), 1)

    def compute_answer_metrics(self, answer: str, contexts: list) -> Dict[str, float]:
        """Token-overlap based Precision/Recall/F1/Accuracy for THIS answer, comparing
        generated answer words against retrieved context words (SQuAD-style F1 metric,
        adapted as a per-answer groundedness proxy since classification metrics don't
        directly apply to free-text generation without a ground-truth reference)."""
        import re
        from collections import Counter

        def tokenize(text):
            return re.findall(r"\\w+", text.lower())

        answer_tokens = tokenize(answer)
        context_text = " ".join(
            c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            for c in contexts
        )
        context_tokens = tokenize(context_text)

        if not answer_tokens or not context_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        answer_counts = Counter(answer_tokens)
        context_counts = Counter(context_tokens)
        common = answer_counts & context_counts
        num_common = sum(common.values())

        precision = num_common / len(answer_tokens)
        recall = num_common / len(context_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = 1.0 if f1 >= 0.3 else 0.0

        return {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "accuracy": round(accuracy * 100, 1)
        }'''

if old_anchor in content:
    content = content.replace(old_anchor, new_block, 1)
    with open("rag_engine.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Added compute_answer_metrics method.")
else:
    print("WARNING: anchor not found — send me `grep -n 'avg_entailment' rag_engine.py` instead.")
