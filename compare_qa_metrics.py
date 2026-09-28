import time
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
        "An unfair trade practice is a trade practice that adopts unfair or deceptive methods to promote the sale, use, or supply of goods or services, including false representations and misleading claims.",
    "What is a misleading advertisement?":
        "A misleading advertisement falsely describes goods or services, gives a false guarantee, or conveys a representation likely to mislead consumers about the nature, quality, or standard of goods or services.",
    "What are the powers of the State Commission?":
        "The State Commission entertains complaints where the value exceeds the District Commission's pecuniary limit but is within the prescribed state limit, hears appeals against District Commission orders, and exercises revisional jurisdiction.",
    "What remedies are available to a consumer?":
        "Remedies include removal of defects, replacement of goods, refund of price, compensation for loss or injury, discontinuation of unfair practices, withdrawal of hazardous goods, and payment of costs.",
    "What is meant by deficiency in service?":
        "Deficiency in service means any fault, imperfection, shortcoming, or inadequacy in the quality, nature, or manner of performance required to be maintained under law or a contract.",
    "What does Article 21 of the Constitution protect?":
        "Article 21 protects the right to life and personal liberty, stating that no person shall be deprived of these except according to procedure established by law.",
    "Explain the writ jurisdiction under Article 32.":
        "Article 32 allows individuals to directly approach the Supreme Court for enforcement of Fundamental Rights, and empowers the Court to issue writs including habeas corpus, mandamus, prohibition, quo warranto, and certiorari.",
    "How does Article 246 distribute legislative powers?":
        "Article 246 distributes legislative power between Parliament and State Legislatures through the Union, State, and Concurrent Lists in the Seventh Schedule, giving Parliament exclusive power over the Union List, States over the State List, and both over the Concurrent List.",
    "What are the Directive Principles of State Policy?":
        "The Directive Principles, found in Part IV of the Constitution, are guidelines for the government to establish social and economic democracy; they are not enforceable in court but are fundamental to governance.",
    "What is the significance of the Preamble?":
        "The Preamble declares India a sovereign, socialist, secular, democratic republic, states the objectives of justice, liberty, equality, and fraternity, and serves as a guide to interpreting the Constitution.",
    "What is Article 14 about equality before law?":
        "Article 14 guarantees that the State shall not deny any person equality before the law or the equal protection of the laws within the territory of India.",
    "What powers does the President have under the Constitution?":
        "The President holds executive powers such as appointing the Prime Minister, ministers, judges, and Governors, legislative powers to summon Parliament and give assent to bills, the power to issue ordinances, and is the Supreme Commander of the armed forces.",
    "What is Article 19 and what freedoms does it protect?":
        "Article 19 guarantees six freedoms: speech and expression, assembly, association, movement, residence, and profession or trade, subject to reasonable restrictions.",
    "How is the Supreme Court's jurisdiction defined?":
        "The Supreme Court has original jurisdiction over disputes between the Union and states, appellate jurisdiction over civil, criminal, and constitutional matters, writ jurisdiction under Article 32, and advisory jurisdiction under Article 143.",
    "What is the procedure to amend the Constitution?":
        "Under Article 368, a constitutional amendment bill must be passed by a special majority in each House of Parliament, and certain amendments additionally require ratification by at least half of the state legislatures.",
    "What are Fundamental Rights under Part III?":
        "Part III guarantees the right to equality, right to freedom, right against exploitation, right to freedom of religion, cultural and educational rights, and the right to constitutional remedies.",
    "What is the role of the Governor under the Constitution?":
        "The Governor is the head of the state executive, appoints the Chief Minister and Council of Ministers, gives assent to state bills, can reserve bills for the President, and holds certain discretionary powers.",
}

MODELS_TO_TEST = ["llama3.2:1b", "llama3.2:3b"]
results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
    engine = LegalRAGEngine(model_name=model_name)

    accs, precs, recs, f1s, grounds, times = [], [], [], [], [], []

    for q, ref in REFERENCE_ANSWERS.items():
        start = time.time()
        contexts, is_confident, min_distance, predicted_domain = engine.retrieve(q)
        if not is_confident:
            continue

        full_answer = ""
        for token in engine.generate_stream(q, contexts):
            full_answer += token

        elapsed = time.time() - start
        groundedness = engine.check_groundedness(full_answer, contexts)
        qa_metrics = engine.compute_qa_metrics(full_answer, ref)

        accs.append(qa_metrics["accuracy"])
        precs.append(qa_metrics["precision"])
        recs.append(qa_metrics["recall"])
        f1s.append(qa_metrics["f1"])
        grounds.append(groundedness)
        times.append(elapsed)

        print(f"  Q: {q[:50]:<50} | f1={qa_metrics['f1']:5.1f}% | grounded={groundedness:5.1f}% | t={elapsed:.2f}s")

    results[model_name] = {
        "accuracy": np.mean(accs) if accs else 0,
        "precision": np.mean(precs) if precs else 0,
        "recall": np.mean(recs) if recs else 0,
        "f1": np.mean(f1s) if f1s else 0,
        "groundedness": np.mean(grounds) if grounds else 0,
        "avg_time": np.mean(times) if times else 0,
    }

print(f"\n{'='*70}\nSUMMARY — QA Metrics vs Reference Answers\n{'='*70}")
print(f"{'Model':<15}{'Accuracy':<11}{'Precision':<11}{'Recall':<11}{'F1':<11}{'Groundedness':<13}{'AvgTime':<9}")
for model_name, m in results.items():
    print(f"{model_name:<15}{m['accuracy']:<10.1f}%{m['precision']:<10.1f}%{m['recall']:<10.1f}%{m['f1']:<10.1f}%{m['groundedness']:<12.1f}%{m['avg_time']:<8.2f}s")

metrics_to_plot = ["accuracy", "precision", "recall", "f1", "groundedness"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score", "Groundedness"]
x = np.arange(len(metrics_to_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#3b82f6", "#c5a059"]
for i, model_name in enumerate(MODELS_TO_TEST):
    vals = [results[model_name][m] for m in metrics_to_plot]
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=model_name, color=colors[i])
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')

ax.set_ylabel("Score (%)")
ax.set_title("RAG QA Performance vs Reference Answers\n(Token-overlap Precision/Recall/F1 against gold answers)")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/qa_metrics_comparison.png", dpi=150)
plt.close()
print("\nSaved: outputs/qa_metrics_comparison.png")
