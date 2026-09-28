"""
semantic_metrics.py — semantic (embedding-based) QA scoring.
Standalone sanity check before this gets wired into compare_qa_metrics.py.
"""

import re
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def semantic_qa_metrics(generated_answer: str, reference_answer: str) -> dict:
    gen_sents = split_sentences(generated_answer)
    ref_sents = split_sentences(reference_answer)
    if not gen_sents or not ref_sents:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gen_vecs = embedder.embed_documents(gen_sents)
    ref_vecs = embedder.embed_documents(ref_sents)

    precision = float(np.mean([max(cosine_sim(g, r) for r in ref_vecs) for g in gen_vecs]))
    recall = float(np.mean([max(cosine_sim(r, g) for g in gen_vecs) for r in ref_vecs]))
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    reference = "The six consumer rights are the right to safety, the right to be informed, the right to choose, the right to be heard, the right to redress, and the right to consumer education."
    generated_good = "Under the Consumer Protection Act, consumers are guaranteed six rights: safety, information, choice, being heard, redress, and consumer education."
    generated_bad = "The Constitution of India establishes fundamental rights including equality and freedom of speech."

    print("Reference vs. GOOD generated answer:")
    print(semantic_qa_metrics(generated_good, reference))
    print()
    print("Reference vs. off-topic BAD generated answer:")
    print(semantic_qa_metrics(generated_bad, reference))
