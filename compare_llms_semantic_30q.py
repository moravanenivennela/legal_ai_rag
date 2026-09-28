import time
import re
import numpy as np
import matplotlib.pyplot as plt
import ollama
from rag_engine import LegalRAGEngine

REFERENCE_ANSWERS = {
    "What are the six consumer rights under the Consumer Protection Act?":
        "The six consumer rights are the right to safety, right to be informed, right to choose, right to be heard, right to seek redressal, and right to consumer education.",
    "What is the pecuniary jurisdiction of the District Commission?":
        "Under the Consumer Protection Act 2019, the District Commission has jurisdiction over complaints where the value of goods or services paid as consideration does not exceed one crore rupees.",
    "What is Product Liability under the 2019 Act?":
        "Product liability makes a manufacturer, product seller, or product service provider liable to compensate a consumer for harm caused by a defective product or deficient service related to the product.",
    "What is an unfair trade practice under consumer law?":
        "An unfair trade practice involves adopting deceptive methods for promoting the sale of goods or services, such as false representations about quality, standard, or usefulness.",
    "What is a misleading advertisement?":
        "A misleading advertisement is one that falsely describes a product or service, gives a false guarantee, or conveys an express or implied representation that would constitute an unfair trade practice.",
    "What are the powers of the State Commission?":
        "The State Commission can entertain complaints where the value of goods or services exceeds the District Commission's limit but is within its own pecuniary jurisdiction, and can hear appeals from District Commissions.",
    "What remedies are available to a consumer?":
        "Remedies include removal of defects, replacement of goods, refund of price paid, compensation for loss or injury, discontinuation of unfair trade practices, and payment of adequate costs.",
    "What is meant by deficiency in service?":
        "Deficiency in service means any fault, imperfection, shortcoming, or inadequacy in the quality, nature, or manner of performance of a service.",
    "What is the role of the Central Consumer Protection Authority?":
        "The CCPA regulates matters relating to violation of consumer rights, unfair trade practices, and false or misleading advertisements, and can take suo motu action, recall goods, and impose penalties.",
    "What is the jurisdiction of the National Commission?":
        "The National Commission has jurisdiction over complaints where the value of goods or services exceeds ten crore rupees, and hears appeals from State Commissions.",
    "How does the Act define a defect in goods?":
        "A defect means any fault, imperfection, or shortcoming in the quality, quantity, potency, purity, or standard required by law or claimed by the trader.",
    "What is the procedure for filing a consumer complaint?":
        "A consumer complaint can be filed in writing to the appropriate Commission along with supporting documents, and can also be filed electronically or through mediation.",
    "What penalties exist for false advertising?":
        "Penalties for false or misleading advertisement include fines up to ten lakh rupees and imprisonment up to two years for repeated offenses.",
    "Who can file a complaint under the Consumer Protection Act?":
        "A complaint can be filed by a consumer, a registered consumer association, the Central or State Government, or one or more consumers on behalf of a group with the same interest.",
    "What is the composition of the State Council?":
        "The State Consumer Protection Council consists of the Minister in-charge of consumer affairs as Chairperson and other official and non-official members as prescribed.",
    "What does Article 21 of the Constitution protect?":
        "Article 21 protects the right to life and personal liberty, stating no person shall be deprived of life or personal liberty except according to procedure established by law.",
    "Explain the writ jurisdiction under Article 32.":
        "Article 32 grants the Supreme Court power to issue writs including habeas corpus, mandamus, prohibition, quo warranto, and certiorari for enforcement of Fundamental Rights.",
    "How does Article 246 distribute legislative powers?":
        "Article 246 distributes legislative powers between Parliament and State Legislatures through the Union List, State List, and Concurrent List in the Seventh Schedule.",
    "What are the Directive Principles of State Policy?":
        "The Directive Principles are guidelines in Part IV of the Constitution for the State to follow while framing laws and policies, aimed at establishing social and economic justice.",
    "What is the significance of the Preamble?":
        "The Preamble declares India to be a sovereign, socialist, secular, democratic republic and sets out the objectives of justice, liberty, equality, and fraternity.",
    "What is Article 14 about equality before law?":
        "Article 14 guarantees equality before the law and equal protection of the laws to all persons within the territory of India.",
    "What powers does the President have under the Constitution?":
        "The President is the head of state with powers including appointing the Prime Minister and other officials, granting pardons, and acting as the Supreme Commander of the armed forces, largely on the advice of the Council of Ministers.",
    "What are Fundamental Rights under Part III?":
        "Fundamental Rights under Part III include the right to equality, right to freedom, right against exploitation, right to freedom of religion, cultural and educational rights, and the right to constitutional remedies.",
    "Explain the concept of judicial review under the Constitution.":
        "Judicial review is the power of courts to examine the constitutionality of legislative and executive actions and declare them void if they violate the Constitution.",
    "What fundamental duties does a citizen have?":
        "Fundamental Duties under Article 51A include respecting the Constitution, promoting harmony, protecting the environment, and safeguarding public property.",
    "How is the Union List different from the State List?":
        "The Union List contains subjects on which only Parliament can legislate, while the State List contains subjects on which only State Legislatures can generally legislate.",
    "What is Article 19 and what freedoms does it protect?":
        "Article 19 protects six freedoms including freedom of speech and expression, assembly, association, movement, residence, and profession, subject to reasonable restrictions.",
    "How is the Supreme Court's jurisdiction defined?":
        "The Supreme Court has original, appellate, and advisory jurisdiction, and is the final court of appeal and guardian of the Constitution.",
    "What is the procedure to amend the Constitution?":
        "Constitutional amendments under Article 368 require a special majority in Parliament, and certain amendments also require ratification by at least half the state legislatures.",
    "What is the difference between Article 32 and Article 226?":
        "Article 32 allows only the Supreme Court to issue writs for Fundamental Rights, while Article 226 allows High Courts to issue writs for both Fundamental Rights and other legal rights, giving High Courts wider jurisdiction.",
}

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def semantic_f1(embedder, generated_answer, reference_answer):
    gen_sents = split_sentences(generated_answer)
    ref_sents = split_sentences(reference_answer)
    if not gen_sents or not ref_sents:
        return 0.0, 0.0, 0.0
    gen_embs = np.array(embedder.embed_documents(gen_sents))
    ref_embs = np.array(embedder.embed_documents(ref_sents))
    def cos_sim_matrix(A, B):
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T
    sim_matrix = cos_sim_matrix(gen_embs, ref_embs)
    precision = sim_matrix.max(axis=1).mean()
    recall = sim_matrix.max(axis=0).mean()
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

ACCURACY_THRESHOLD = 0.65

installed = [m.model for m in ollama.list().models]
print("Installed Ollama models:", installed)

MODELS_TO_TEST = [m for m in ["llama3.2:1b", "gemma2:2b", "gemma3:4b", "llama3.2:3b"] if any(m in i for i in installed)]
print("Testing models:", MODELS_TO_TEST)
print(f"Total questions: {len(REFERENCE_ANSWERS)}")

results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
    engine = LegalRAGEngine(model_name=model_name)

    precs, recs, f1s, accs, grounds, times = [], [], [], [], [], []

    for q, ref_answer in REFERENCE_ANSWERS.items():
        start = time.time()
        dense_hits, min_distance = engine.dense_search(q, k=10)
        sparse_hits = engine.sparse_search(q, k=10)
        fused = engine.reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=10)
        contexts = engine.rerank(q, fused, top_n=6)
        if not contexts:
            continue

        full_answer = ""
        for token in engine.generate_stream(q, contexts):
            full_answer += token
        elapsed = time.time() - start

        groundedness = engine.check_groundedness(full_answer, contexts)
        p, r, f1 = semantic_f1(engine.embedding_fn, full_answer, ref_answer)
        acc = 1.0 if f1 >= ACCURACY_THRESHOLD else 0.0

        precs.append(p * 100); recs.append(r * 100); f1s.append(f1 * 100)
        accs.append(acc * 100); grounds.append(groundedness); times.append(elapsed)

        print(f"  Q: {q[:50]:<50} | P={p*100:5.1f}% R={r*100:5.1f}% F1={f1*100:5.1f}% | grounded={groundedness:5.1f}% | t={elapsed:.1f}s")

    results[model_name] = {
        "accuracy": np.mean(accs) if accs else 0,
        "precision": np.mean(precs) if precs else 0,
        "recall": np.mean(recs) if recs else 0,
        "f1": np.mean(f1s) if f1s else 0,
        "groundedness": np.mean(grounds) if grounds else 0,
        "avg_time": np.mean(times) if times else 0,
    }

print(f"\n\n{'='*80}")
print(f"SUMMARY — Semantic RAG Performance Across LLM Models (n={len(REFERENCE_ANSWERS)} questions)")
print(f"{'='*80}")
print(f"{'Model':<16}{'Accuracy':<11}{'Precision':<11}{'Recall':<11}{'F1':<11}{'Groundedness':<14}{'AvgTime':<9}")
for model_name, m in results.items():
    print(f"{model_name:<16}{m['accuracy']:<10.1f}%{m['precision']:<10.1f}%{m['recall']:<10.1f}%"
          f"{m['f1']:<10.1f}%{m['groundedness']:<13.1f}%{m['avg_time']:<8.2f}s")

metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
x = np.arange(len(metrics_to_plot))
n_models = len(MODELS_TO_TEST)
width = 0.8 / max(n_models, 1)

fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#3b82f6", "#c5a059", "#10b981", "#9333EA"]
for i, model_name in enumerate(MODELS_TO_TEST):
    vals = [results[model_name][m] for m in metrics_to_plot]
    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width, label=model_name, color=colors[i % len(colors)])
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title(f"RAG Performance Across LLM Models — Expanded Evaluation (n={len(REFERENCE_ANSWERS)} questions)\nSame hybrid retrieval pipeline (top-6 reranked chunks), different generation models",
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 110)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/rag_performance_across_llms_30q.png", dpi=150)
plt.close()
print("\nSaved: outputs/rag_performance_across_llms_30q.png")
