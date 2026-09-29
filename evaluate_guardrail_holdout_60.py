from rag_engine import LegalRAGEngine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# FRESH UNSEEN HOLDOUT SET — 60 QUESTIONS
# 20 Constitution
# 20 Consumer Protection
# 20 Out-of-Domain
# ============================================================

questions = [

    # ========================================================
    # CONSTITUTION — 20
    # ========================================================

    ("What constitutional provision deals with equality of opportunity in public employment?",
     "in_domain"),

    ("What does the Constitution provide regarding abolition of untouchability?",
     "in_domain"),

    ("Which constitutional provision protects freedom of speech and expression?",
     "in_domain"),

    ("What is the constitutional protection against arrest and detention?",
     "in_domain"),

    ("What does Article 32 provide to citizens?",
     "in_domain"),

    ("What are the constitutional safeguards for educational and cultural rights?",
     "in_domain"),

    ("What is the purpose of the Directive Principles of State Policy?",
     "in_domain"),

    ("What does the Constitution say about separation of the judiciary from the executive?",
     "in_domain"),

    ("What constitutional provision deals with equal justice and free legal aid?",
     "in_domain"),

    ("What does the Constitution provide concerning village panchayats?",
     "in_domain"),

    ("What is the constitutional position regarding the Uniform Civil Code?",
     "in_domain"),

    ("Which constitutional provision concerns protection of the environment?",
     "in_domain"),

    ("What duties does the Constitution impose on citizens regarding the environment?",
     "in_domain"),

    ("What constitutional provisions deal with the office of the President of India?",
     "in_domain"),

    ("What is the constitutional role of the Council of Ministers in relation to the President?",
     "in_domain"),

    ("What does the Constitution provide about the appointment of the Governor?",
     "in_domain"),

    ("What constitutional provisions govern the Supreme Court of India?",
     "in_domain"),

    ("What does the Constitution say about the Comptroller and Auditor-General of India?",
     "in_domain"),

    ("What is the constitutional procedure for amending the Constitution?",
     "in_domain"),

    ("What does the Constitution provide regarding financial emergency?",
     "in_domain"),


    # ========================================================
    # CONSUMER PROTECTION — 20
    # ========================================================

    ("Who can be considered a consumer under the Consumer Protection Act, 2019?",
     "in_domain"),

    ("What is meant by a defect in goods under consumer protection law?",
     "in_domain"),

    ("What is meant by deficiency in services?",
     "in_domain"),

    ("What constitutes an unfair contract under the Consumer Protection Act?",
     "in_domain"),

    ("What remedies can a consumer commission grant to a complainant?",
     "in_domain"),

    ("How can a consumer complaint be filed electronically?",
     "in_domain"),

    ("What is the jurisdiction of the District Consumer Disputes Redressal Commission?",
     "in_domain"),

    ("What is the role of the State Consumer Disputes Redressal Commission?",
     "in_domain"),

    ("What is the role of the National Consumer Disputes Redressal Commission?",
     "in_domain"),

    ("What powers does the Central Consumer Protection Authority have?",
     "in_domain"),

    ("How does the Act address false or misleading advertisements?",
     "in_domain"),

    ("What is product liability under the Consumer Protection Act, 2019?",
     "in_domain"),

    ("Who may be held responsible for harm caused by a defective product?",
     "in_domain"),

    ("What is an unfair trade practice under consumer protection law?",
     "in_domain"),

    ("What is a restrictive trade practice?",
     "in_domain"),

    ("How does the Consumer Protection Act deal with e-commerce transactions?",
     "in_domain"),

    ("What is the purpose of consumer mediation?",
     "in_domain"),

    ("Who may participate in the consumer mediation process?",
     "in_domain"),

    ("What information must be provided to consumers in an e-commerce transaction?",
     "in_domain"),

    ("What penalties may apply for manufacturing or selling spurious goods?",
     "in_domain"),


    # ========================================================
    # OUT-OF-DOMAIN — 20
    # ========================================================

    ("What are the filing requirements under the Companies Act, 2013?",
     "out_of_domain"),

    ("How is income tax calculated for an individual under Indian tax law?",
     "out_of_domain"),

    ("What is the procedure for filing a GST return?",
     "out_of_domain"),

    ("What are the eligibility requirements for a trademark registration?",
     "out_of_domain"),

    ("How does the Motor Vehicles Act regulate driving licences?",
     "out_of_domain"),

    ("What is the procedure for obtaining a passport in India?",
     "out_of_domain"),

    ("What are the provisions of the Indian Contract Act regarding consideration?",
     "out_of_domain"),

    ("How does the Negotiable Instruments Act deal with cheque dishonour?",
     "out_of_domain"),

    ("What are the requirements for registration under the Copyright Act?",
     "out_of_domain"),

    ("How does the Patents Act protect inventions?",
     "out_of_domain"),

    ("What is the process for registering a company in India?",
     "out_of_domain"),

    ("What are the rules governing income tax deductions?",
     "out_of_domain"),

    ("How are goods and services tax refunds processed?",
     "out_of_domain"),

    ("What are the requirements for obtaining a driving licence?",
     "out_of_domain"),

    ("How does the Information Technology Act address cyber offences?",
     "out_of_domain"),

    ("What are the provisions governing industrial disputes?",
     "out_of_domain"),

    ("How does the Insolvency and Bankruptcy Code handle corporate insolvency?",
     "out_of_domain"),

    ("What are the regulations governing securities markets in India?",
     "out_of_domain"),

    ("What does the Right to Information Act provide about access to government records?",
     "out_of_domain"),

    ("What are the provisions of the Environmental Protection Act concerning pollution control?",
     "out_of_domain"),
]


def main():

    print("=" * 70)
    print("FRESH UNSEEN HOLDOUT — RETRIEVAL GUARDRAIL")
    print("=" * 70)
    print(f"Total questions: {len(questions)}")
    print("Constitution: 20")
    print("Consumer Protection: 20")
    print("Out-of-domain: 20")
    print("=" * 70)

    engine = LegalRAGEngine()

    y_true = []
    y_pred = []

    results = []

    for i, (question, expected) in enumerate(questions, start=1):

        try:
            contexts, allowed, distance, predicted = engine.retrieve(question)

            y_true.append(expected)

            # Convert the engine's 3-class output into the
            # binary guardrail evaluation used by this experiment.
            if predicted in ("constitution", "consumer_protection"):
                predicted_binary = "in_domain"
            elif predicted == "out_of_domain":
                predicted_binary = "out_of_domain"
            else:
                predicted_binary = "out_of_domain"

            y_pred.append(predicted_binary)

            correct = expected == predicted_binary

            results.append({
                "id": i,
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "distance": distance,
            })

            status = "OK" if correct else "ERROR"

            print(
                f"[{i:02d}/60] {status} | "
                f"Expected={expected:<13} "
                f"Predicted={predicted:<13} | "
                f"{question}"
            )

        except Exception as e:

            y_true.append(expected)
            y_pred.append("ERROR")

            results.append({
                "id": i,
                "question": question,
                "expected": expected,
                "predicted": "ERROR",
                "correct": False,
                "distance": None,
            })

            print(f"[{i:02d}/60] ERROR | {question}")
            print(f"       {e}")

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="in_domain",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="in_domain",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="in_domain",
        zero_division=0
    )

    labels = ["out_of_domain", "in_domain"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("\n")
    print("=" * 70)
    print("FRESH UNSEEN HOLDOUT — FINAL RESULTS")
    print("=" * 70)

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    errors = [r for r in results if not r["correct"]]

    print("\n")
    print("=" * 70)
    print(f"ERROR ANALYSIS — {len(errors)} ERRORS")
    print("=" * 70)

    if not errors:
        print("NO ERRORS — ALL 60 QUESTIONS CLASSIFIED CORRECTLY.")
    else:
        for r in errors:
            print(f"\n#{r['id']}")
            print(f"Question : {r['question']}")
            print(f"Expected : {r['expected']}")
            print(f"Predicted: {r['predicted']}")
            print(f"Distance : {r['distance']}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Fresh 60-Question Guardrail Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(range(len(labels)), labels, rotation=20)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix_guardrail_holdout_60.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Metrics chart
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    metric_values = [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(metric_names, metric_values)
    plt.ylim(0, 105)
    plt.ylabel("Percentage")
    plt.title("Fresh 60-Question Guardrail Evaluation")

    for i, value in enumerate(metric_values):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")

    plt.tight_layout()

    metrics_path = output_dir / "evaluation_metrics_guardrail_holdout_60.png"
    plt.savefig(metrics_path, dpi=200)
    plt.close()

    # Save text report
    report_path = output_dir / "guardrail_holdout_60_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("FRESH UNSEEN HOLDOUT — RETRIEVAL GUARDRAIL\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total questions: {len(questions)}\n")
        f.write("Constitution: 20\n")
        f.write("Consumer Protection: 20\n")
        f.write("Out-of-domain: 20\n\n")

        f.write(f"Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall:    {recall * 100:.2f}%\n")
        f.write(f"F1-score:  {f1 * 100:.2f}%\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("ERRORS:\n")

        if not errors:
            f.write("None\n")
        else:
            for r in errors:
                f.write(f"\n#{r['id']}\n")
                f.write(f"Question: {r['question']}\n")
                f.write(f"Expected: {r['expected']}\n")
                f.write(f"Predicted: {r['predicted']}\n")
                f.write(f"Distance: {r['distance']}\n")

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(cm_path)
    print(metrics_path)
    print(report_path)


if __name__ == "__main__":
    main()
