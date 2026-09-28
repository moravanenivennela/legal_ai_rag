import sys, time, re, json, os
import numpy as np
from rag_engine import LegalRAGEngine

if len(sys.argv) < 2:
    print("Usage: python run_single_model_eval.py <model_name>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
RESULTS_FILE = "llm_results_incremental.json"

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
        return None  # flagged as failed, not zero
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

print(f"Evaluating model: {MODEL_NAME} | n={len(REFERENCE_ANSWERS)} questions")
engine = LegalRAGEngine(model_name=MODEL_NAME)

precs, recs, f1s, accs, grounds, times, failed_questions = [], [], [], [], [], [], []

for q, ref_answer in REFERENCE_ANSWERS.items():
    try:
        start = time.time()
        dense_hits, min_distance = engine.dense_search(q, k=10)
        sparse_hits = engine.sparse_search(q, k=10)
        fused = engine.reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=10)
        contexts = engine.rerank(q, fused, top_n=6)
        if not contexts:
            failed_questions.append(q)
            continue

        full_answer = ""
        for token in engine.generate_stream(q, contexts):
            full_answer += token
        elapsed = time.time() - start

        if not full_answer.strip():
            print(f"  SKIPPED (empty answer): {q[:50]}")
            failed_questions.append(q)
            continue

        groundedness = engine.check_groundedness(full_answer, contexts)
        result = semantic_f1(engine.embedding_fn, full_answer, ref_answer)
        if result is None:
            print(f"  SKIPPED (no valid sentences): {q[:50]}")
            failed_questions.append(q)
            continue
        p, r, f1 = result
        acc = 1.0 if f1 >= ACCURACY_THRESHOLD else 0.0

        precs.append(p*100); recs.append(r*100); f1s.append(f1*100)
        accs.append(acc*100); grounds.append(groundedness); times.append(elapsed)
        print(f"  {q[:50]:<50} F1={f1*100:5.1f}% t={elapsed:.1f}s")
    except Exception as e:
        print(f"  ERROR on question '{q[:40]}': {e}")
        failed_questions.append(q)

result_entry = {
    "accuracy": float(np.mean(accs)) if accs else None,
    "precision": float(np.mean(precs)) if precs else None,
    "recall": float(np.mean(recs)) if recs else None,
    "f1": float(np.mean(f1s)) if f1s else None,
    "groundedness": float(np.mean(grounds)) if grounds else None,
    "avg_time": float(np.mean(times)) if times else None,
    "n_completed": len(accs),
    "n_failed": len(failed_questions),
    "failed_questions": failed_questions,
}

# Load existing results (if any) and merge
all_results = {}
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
all_results[MODEL_NAME] = result_entry

with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n=== {MODEL_NAME} RESULT ===")
print(f"Completed: {result_entry['n_completed']}/{len(REFERENCE_ANSWERS)}, Failed: {result_entry['n_failed']}")
if result_entry["accuracy"] is not None:
    print(f"Accuracy={result_entry['accuracy']:.1f}% Precision={result_entry['precision']:.1f}% "
          f"Recall={result_entry['recall']:.1f}% F1={result_entry['f1']:.1f}% "
          f"Groundedness={result_entry['groundedness']:.1f}% AvgTime={result_entry['avg_time']:.1f}s")
print(f"\nSaved to {RESULTS_FILE}")
