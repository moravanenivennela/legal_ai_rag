from rag_engine import LegalRAGEngine
import csv
import time
from pathlib import Path

TESTS = [
    ("What does Article 14 guarantee?", "constitution_of_india.pdf"),
    ("What does Article 21 protect?", "constitution_of_india.pdf"),
    ("What does Article 32 provide?", "constitution_of_india.pdf"),
    ("What does Article 43 provide?", "constitution_of_india.pdf"),
    ("What does Article 44 provide?", "constitution_of_india.pdf"),
    ("What are Fundamental Duties?", "constitution_of_india.pdf"),
    ("What are the Directive Principles?", "constitution_of_india.pdf"),
    ("What is the Seventh Schedule?", "constitution_of_india.pdf"),
    ("What is the Concurrent List?", "constitution_of_india.pdf"),
    ("What is President's Rule?", "constitution_of_india.pdf"),
    ("What are the powers of the Supreme Court?", "constitution_of_india.pdf"),
    ("What is the constitutional position of the Election Commission?", "constitution_of_india.pdf"),
    ("What is the Finance Commission?", "constitution_of_india.pdf"),
    ("What is the procedure for constitutional amendment?", "constitution_of_india.pdf"),
    ("What are the constitutional emergency provisions?", "constitution_of_india.pdf"),
    ("What is a consumer under the Consumer Protection Act, 2019?", "consumer_protection_act_2019.pdf"),
    ("What is a defect in goods?", "consumer_protection_act_2019.pdf"),
    ("What is deficiency in service?", "consumer_protection_act_2019.pdf"),
    ("What is an unfair trade practice?", "consumer_protection_act_2019.pdf"),
    ("What is a restrictive trade practice?", "consumer_protection_act_2019.pdf"),
    ("What is product liability?", "consumer_protection_act_2019.pdf"),
    ("What is an unfair contract?", "consumer_protection_act_2019.pdf"),
    ("What is the CCPA?", "consumer_protection_act_2019.pdf"),
    ("What is consumer mediation?", "consumer_protection_act_2019.pdf"),
    ("What is the role of the District Consumer Commission?", "consumer_protection_act_2019.pdf"),
    ("What is the role of the State Consumer Commission?", "consumer_protection_act_2019.pdf"),
    ("What is the role of the National Consumer Commission?", "consumer_protection_act_2019.pdf"),
    ("What is product liability of a manufacturer?", "consumer_protection_act_2019.pdf"),
    ("What is product liability of a seller?", "consumer_protection_act_2019.pdf"),
    ("What remedies can a consumer commission provide?", "consumer_protection_act_2019.pdf"),
]


def collect_answer(engine, question, context):
    return "".join(
        engine.generate_stream(
            question,
            [context],
            language="English",
            eli5=False
        )
    ).strip()


def main():
    engine = LegalRAGEngine()
    rows = []

    print("=" * 70)
    print("TOP-1 CONTEXT GENERATION EXPERIMENT")
    print("=" * 70)

    for i, (question, expected_source) in enumerate(TESTS, 1):
        print(f"\n[{i:02d}/30] {question}")

        start = time.perf_counter()

        try:
            hits, retrieval_success, min_distance, predicted_domain = \
                engine.retrieve(question)

            if not retrieval_success or not hits:
                print("  Retrieval failed")
                continue

            top1 = hits[0]

            source = top1.get("metadata", {}).get("source", "")
            source_match = expected_source == source

            answer = collect_answer(engine, question, top1)

            groundedness = engine.check_groundedness(
                answer,
                [top1]
            )

            latency = time.perf_counter() - start

            print(f"  Source:       {source}")
            print(f"  Source match: {source_match}")
            print(f"  Groundedness: {groundedness:.1f}%")
            print(f"  Latency:      {latency:.2f}s")
            print(f"  Answer:       {answer[:250]}")

            rows.append({
                "question": question,
                "expected_source": expected_source,
                "source": source,
                "source_match": source_match,
                "predicted_domain": predicted_domain,
                "groundedness": groundedness,
                "latency_seconds": round(latency, 2),
                "answer": answer
            })

        except Exception as e:
            print(f"  ERROR: {e}")

    if not rows:
        print("No successful results.")
        return

    avg_groundedness = sum(
        r["groundedness"] for r in rows
    ) / len(rows)

    grounded_70 = sum(
        r["groundedness"] >= 70 for r in rows
    ) / len(rows)

    source_accuracy = sum(
        r["source_match"] for r in rows
    ) / len(rows)

    avg_latency = sum(
        r["latency_seconds"] for r in rows
    ) / len(rows)

    print()
    print("=" * 70)
    print("TOP-1 EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Successful answers:       {len(rows)}/30")
    print(f"Source match:             {source_accuracy * 100:.2f}%")
    print(f"Average groundedness:     {avg_groundedness:.2f}%")
    print(f"Groundedness >= 70%:      {grounded_70 * 100:.2f}%")
    print(f"Average latency:          {avg_latency:.2f}s")

    Path("outputs").mkdir(exist_ok=True)

    with open(
        "outputs/TOP1_GENERATION_EVALUATION.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "expected_source",
                "source",
                "source_match",
                "predicted_domain",
                "groundedness",
                "latency_seconds",
                "answer"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(
        "outputs/TOP1_GENERATION_EVALUATION_REPORT.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("TOP-1 CONTEXT GENERATION EXPERIMENT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Successful answers: {len(rows)}/30\n")
        f.write(f"Source match: {source_accuracy * 100:.2f}%\n")
        f.write(f"Average groundedness: {avg_groundedness:.2f}%\n")
        f.write(f"Groundedness >= 70%: {grounded_70 * 100:.2f}%\n")
        f.write(f"Average latency: {avg_latency:.2f}s\n")

    print("\nSaved:")
    print("outputs/TOP1_GENERATION_EVALUATION.csv")
    print("outputs/TOP1_GENERATION_EVALUATION_REPORT.txt")


if __name__ == "__main__":
    main()
