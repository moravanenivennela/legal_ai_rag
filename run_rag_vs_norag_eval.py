import sys, time, re, json, os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from rag_engine import LegalRAGEngine

if len(sys.argv) < 2:
    print("Usage: python run_rag_vs_norag_eval.py <model_name>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
RESULTS_FILE = "rag_vs_norag_incremental.json"

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
        return None
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
print(f"RAG vs No-RAG evaluation for: {MODEL_NAME} | n={len(REFERENCE_ANSWERS)} questions")
engine = LegalRAGEngine(model_name=MODEL_NAME)

rag_metrics = {"precs": [], "recs": [], "f1s": [], "accs": [], "grounds": [], "times": []}
norag_metrics = {"precs": [], "recs": [], "f1s": [], "accs": [], "times": []}

for q, ref_answer in REFERENCE_ANSWERS.items():
    # --- WITH RAG ---
    try:
        start = time.time()
        dense_hits, min_distance = engine.dense_search(q, k=10)
        sparse_hits = engine.sparse_search(q, k=10)
        fused = engine.reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=10)
        contexts = engine.rerank(q, fused, top_n=6)
        rag_answer = ""
        for token in engine.generate_stream(q, contexts):
            rag_answer += token
        rag_time = time.time() - start
        groundedness = engine.check_groundedness(rag_answer, contexts) if contexts else 0.0
        result = semantic_f1(engine.embedding_fn, rag_answer, ref_answer)
        if result:
            p, r, f1 = result
            rag_metrics["precs"].append(p*100); rag_metrics["recs"].append(r*100)
            rag_metrics["f1s"].append(f1*100); rag_metrics["accs"].append(1.0 if f1 >= ACCURACY_THRESHOLD else 0.0)
            rag_metrics["grounds"].append(groundedness); rag_metrics["times"].append(rag_time)
            print(f"  [RAG]    {q[:45]:<45} F1={f1*100:5.1f}% t={rag_time:.1f}s")
    except Exception as e:
        print(f"  [RAG] ERROR: {e}")

    # --- WITHOUT RAG (baseline) ---
    try:
        start = time.time()
        norag_answer = ""
        for token in engine.generate_baseline(q):
            norag_answer += token
        norag_time = time.time() - start
        result = semantic_f1(engine.embedding_fn, norag_answer, ref_answer)
        if result:
            p, r, f1 = result
            norag_metrics["precs"].append(p*100); norag_metrics["recs"].append(r*100)
            norag_metrics["f1s"].append(f1*100); norag_metrics["accs"].append(1.0 if f1 >= ACCURACY_THRESHOLD else 0.0)
            norag_metrics["times"].append(norag_time)
            print(f"  [No-RAG] {q[:45]:<45} F1={f1*100:5.1f}% t={norag_time:.1f}s")
    except Exception as e:
        print(f"  [No-RAG] ERROR: {e}")

result_entry = {
    "rag": {
        "accuracy": float(np.mean(rag_metrics["accs"]))*100 if rag_metrics["accs"] else None,
        "precision": float(np.mean(rag_metrics["precs"])) if rag_metrics["precs"] else None,
        "recall": float(np.mean(rag_metrics["recs"])) if rag_metrics["recs"] else None,
        "f1": float(np.mean(rag_metrics["f1s"])) if rag_metrics["f1s"] else None,
        "groundedness": float(np.mean(rag_metrics["grounds"])) if rag_metrics["grounds"] else None,
        "avg_time": float(np.mean(rag_metrics["times"])) if rag_metrics["times"] else None,
    },
    "norag": {
        "accuracy": float(np.mean(norag_metrics["accs"]))*100 if norag_metrics["accs"] else None,
        "precision": float(np.mean(norag_metrics["precs"])) if norag_metrics["precs"] else None,
        "recall": float(np.mean(norag_metrics["recs"])) if norag_metrics["recs"] else None,
        "f1": float(np.mean(norag_metrics["f1s"])) if norag_metrics["f1s"] else None,
        "avg_time": float(np.mean(norag_metrics["times"])) if norag_metrics["times"] else None,
    },
    "raw_rag_accs": rag_metrics["accs"],
    "raw_norag_accs": norag_metrics["accs"],
}

all_results = {}
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
all_results[MODEL_NAME] = result_entry
with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n=== {MODEL_NAME} SUMMARY ===")
print(f"RAG:    Acc={result_entry['rag']['accuracy']:.1f}% P={result_entry['rag']['precision']:.1f}% R={result_entry['rag']['recall']:.1f}% F1={result_entry['rag']['f1']:.1f}% Ground={result_entry['rag']['groundedness']:.1f}% Time={result_entry['rag']['avg_time']:.1f}s")
print(f"No-RAG: Acc={result_entry['norag']['accuracy']:.1f}% P={result_entry['norag']['precision']:.1f}% R={result_entry['norag']['recall']:.1f}% F1={result_entry['norag']['f1']:.1f}% Time={result_entry['norag']['avg_time']:.1f}s")
print(f"\nSaved to {RESULTS_FILE}")
