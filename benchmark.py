"""
Mini evaluation benchmark — mirrors the paper's accuracy methodology.
Measures RETRIEVAL accuracy: did the correct source article get retrieved?
"""
from rag_engine import LegalRAGEngine
import time

# Add your own QA pairs here. "expected_keyword" = a word/phrase that MUST
# appear in the citation/metadata of a correctly retrieved chunk.
BENCHMARK = [
    {"question": "What are the consumer rights under the Consumer Protection Act?", "expected_keyword": "consumer", "difficulty": "easy"},
    {"question": "What is the pecuniary jurisdiction of the District Commission?", "expected_keyword": "jurisdiction", "difficulty": "medium"},
    {"question": "What is Product Liability under the 2019 Act?", "expected_keyword": "liability", "difficulty": "medium"},
    {"question": "What does Article 21 of the Constitution protect?", "expected_keyword": "21", "difficulty": "easy"},
    {"question": "Explain the writ jurisdiction differences between Article 32 and Article 226.", "expected_keyword": "226", "difficulty": "hard"},
    {"question": "How does Article 246 distribute legislative powers?", "expected_keyword": "246", "difficulty": "hard"},
    {"question": "What is the statutory penalty for driving without a license?", "expected_keyword": None, "difficulty": "guardrail"},
]

def run_benchmark(model_name="llama3.2:3b"):
    engine = LegalRAGEngine(model_name=model_name)
    results = []

    for item in BENCHMARK:
        start = time.time()
        contexts, is_confident, distance = engine.retrieve(item["question"])
        elapsed = time.time() - start

        if item["expected_keyword"] is None:
            correct = not is_confident
        else:
            correct = is_confident and any(
                item["expected_keyword"].lower() in str(c["metadata"]).lower()
                or item["expected_keyword"].lower() in c["text"].lower()
                for c in contexts
            )

        results.append({
            "question": item["question"][:50] + "...",
            "difficulty": item["difficulty"],
            "correct": correct,
            "distance": round(distance, 3),
            "time_sec": round(elapsed, 2)
        })

    total = len(results)
    correct_count = sum(r["correct"] for r in results)
    print(f"\n{'='*60}\nOVERALL ACCURACY: {correct_count}/{total} = {correct_count/total*100:.1f}%\n{'='*60}\n")

    for diff in ["easy", "medium", "hard", "guardrail"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            acc = sum(r["correct"] for r in subset) / len(subset) * 100
            print(f"{diff.upper():10s}: {sum(r['correct'] for r in subset)}/{len(subset)} = {acc:.1f}%")

    print(f"\n{'Question':<55}{'Difficulty':<12}{'Correct':<10}{'Distance':<10}{'Time(s)'}")
    for r in results:
        print(f"{r['question']:<55}{r['difficulty']:<12}{str(r['correct']):<10}{r['distance']:<10}{r['time_sec']}")

if __name__ == "__main__":
    run_benchmark()
