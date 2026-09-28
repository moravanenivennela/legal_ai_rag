import re

with open("rag_engine.py", "r", encoding="utf-8") as f:
    content = f.read()

old_method_pattern = re.compile(
    r'    def check_groundedness\(self, answer: str, contexts: list\) -> float:.*?return round\(max\(0\.0, min\(100\.0, avg_entailment \* 100\)\), 1\)',
    re.DOTALL
)

new_method = '''    def check_groundedness(self, answer: str, contexts: list) -> float:
        if not contexts or not answer.strip():
            return 0.0

        chunk_texts = []
        for c in contexts:
            text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            if text.strip():
                chunk_texts.append(text)

        sentences = [s.strip() for s in answer.replace("\\n", " ").split(".") if len(s.strip()) > 15]

        if not sentences or not chunk_texts:
            return 0.0

        sentence_scores = []
        for sent in sentences:
            pairs = [(chunk, sent) for chunk in chunk_texts]
            scores = self.nli_model.predict(pairs, apply_softmax=True)
            entailment_probs = [s[1] for s in scores]
            best_match = max(entailment_probs)
            sentence_scores.append(best_match)

        avg_entailment = sum(sentence_scores) / len(sentence_scores)
        return round(max(0.0, min(100.0, avg_entailment * 100)), 1)'''

new_content, count = old_method_pattern.subn(new_method, content)

if count == 1:
    with open("rag_engine.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Patched successfully.")
else:
    print(f"WARNING: matched {count} times (expected 1). No changes made.")
