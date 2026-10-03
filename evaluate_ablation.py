from rag_engine import LegalRAGEngine
import numpy as np

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

def source_match(hits, expected):
    if not hits:
        return False
    return hits[0].get("metadata", {}).get("source", "") == expected

def hit_at_k(hits, expected, k):
    return any(
        h.get("metadata", {}).get("source", "") == expected
        for h in hits[:k]
    )

def main():
    engine = LegalRAGEngine()

    results = {
        "Dense": [],
        "BM25": [],
        "Hybrid_RRF": [],
        "Full_Reranker": []
    }

    for i, (question, expected) in enumerate(TESTS, 1):
        print(f"[{i:02d}/30] {question}")

        try:
            qvec = engine.embedding_fn.embed_query(question)

            dense, _ = engine.dense_search(question, k=10)
            sparse = engine.sparse_search(question, k=10)

            fused = engine.reciprocal_rank_fusion(
                dense,
                sparse,
                top_n=8
            )

            reranked = engine.rerank(
                question,
                fused,
                top_n=4
            )

            results["Dense"].append(
                [hit_at_k(dense, expected, k) for k in [1, 3, 4]]
            )

            results["BM25"].append(
                [hit_at_k(sparse, expected, k) for k in [1, 3, 4]]
            )

            results["Hybrid_RRF"].append(
                [hit_at_k(fused, expected, k) for k in [1, 3, 4]]
            )

            results["Full_Reranker"].append(
                [hit_at_k(reranked, expected, k) for k in [1, 3, 4]]
            )

        except Exception as e:
            print("ERROR:", e)

    print()
    print("=" * 75)
    print("RETRIEVAL ABLATION STUDY")
    print("=" * 75)
    print(f"{'Configuration':<22} {'Hit@1':>10} {'Hit@3':>10} {'Hit@4':>10}")
    print("-" * 75)

    for name, rows in results.items():
        if not rows:
            continue

        arr = np.array(rows, dtype=float) * 100

        print(
            f"{name:<22}"
            f"{arr[:,0].mean():>9.2f}%"
            f"{arr[:,1].mean():>9.2f}%"
            f"{arr[:,2].mean():>9.2f}%"
        )

if __name__ == "__main__":
    main()
