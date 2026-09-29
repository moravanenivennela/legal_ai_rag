from rag_engine import LegalRAGEngine
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from pathlib import Path
import csv


# ============================================================
# FINAL BLIND EVALUATION
#
# IMPORTANT:
# This test set must NOT be modified after seeing predictions.
#
# 100 questions:
#   30 Constitution
#   30 Consumer Protection
#   40 Out-of-domain
# ============================================================

TEST_SET = [

    # ========================================================
    # CONSTITUTION — 30
    # ========================================================

    ("What constitutional provision guarantees equality before the law?",
     "constitution"),

    ("Which Article deals with protection against discrimination on specified grounds?",
     "constitution"),

    ("What does the Constitution provide regarding equality of opportunity in public employment?",
     "constitution"),

    ("Which constitutional provision concerns freedom of conscience and religion?",
     "constitution"),

    ("What constitutional protection is available against conviction for offences?",
     "constitution"),

    ("Which provision protects life and personal liberty?",
     "constitution"),

    ("What does the Constitution say about protection of minorities?",
     "constitution"),

    ("Which constitutional provision concerns the right to education?",
     "constitution"),

    ("What is the constitutional status of Fundamental Duties?",
     "constitution"),

    ("What does the Constitution provide regarding citizenship?",
     "constitution"),

    ("Which constitutional provisions deal with Parliament?",
     "constitution"),

    ("What is the constitutional composition of the Rajya Sabha?",
     "constitution"),

    ("What does the Constitution provide concerning the Lok Sabha?",
     "constitution"),

    ("Which constitutional provision deals with the election of the President?",
     "constitution"),

    ("What is the constitutional procedure for impeachment of the President?",
     "constitution"),

    ("What does the Constitution provide regarding the Vice-President?",
     "constitution"),

    ("Which constitutional provisions deal with the Supreme Court?",
     "constitution"),

    ("What does the Constitution provide concerning High Courts?",
     "constitution"),

    ("What is the constitutional position of the Election Commission of India?",
     "constitution"),

    ("Which constitutional provision deals with the Finance Commission?",
     "constitution"),

    ("What does the Constitution provide concerning the Union Public Service Commission?",
     "constitution"),

    ("Which constitutional provisions concern emergency powers?",
     "constitution"),

    ("What is the constitutional effect of a national emergency?",
     "constitution"),

    ("What does the Constitution provide concerning President's Rule?",
     "constitution"),

    ("Which constitutional provision deals with financial emergency?",
     "constitution"),

    ("What is the constitutional procedure for amendment of certain provisions?",
     "constitution"),

    ("What does the Constitution say about distribution of legislative powers?",
     "constitution"),

    ("Which Schedule contains the Union, State and Concurrent Lists?",
     "constitution"),

    ("What does the Constitution provide regarding official language?",
     "constitution"),

    ("What constitutional provisions deal with local self-government?",
     "constitution"),


    # ========================================================
    # CONSUMER PROTECTION — 30
    # ========================================================

    ("What is the definition of consumer under the Consumer Protection Act, 2019?",
     "consumer_protection"),

    ("Who is excluded from the definition of consumer?",
     "consumer_protection"),

    ("What is a complaint under the Consumer Protection Act?",
     "consumer_protection"),

    ("Who can file a consumer complaint?",
     "consumer_protection"),

    ("What is a defect in goods under the Act?",
     "consumer_protection"),

    ("What is deficiency in service under the Act?",
     "consumer_protection"),

    ("What is an unfair trade practice?",
     "consumer_protection"),

    ("What is a restrictive trade practice?",
     "consumer_protection"),

    ("What is product liability?",
     "consumer_protection"),

    ("What is an unfair contract?",
     "consumer_protection"),

    ("What is the Central Consumer Protection Authority?",
     "consumer_protection"),

    ("What functions does the Central Consumer Protection Authority perform?",
     "consumer_protection"),

    ("What powers does the CCPA have regarding misleading advertisements?",
     "consumer_protection"),

    ("What are the consumer commissions established under the Act?",
     "consumer_protection"),

    ("What is the role of the District Consumer Disputes Redressal Commission?",
     "consumer_protection"),

    ("What is the role of the State Consumer Disputes Redressal Commission?",
     "consumer_protection"),

    ("What is the role of the National Consumer Disputes Redressal Commission?",
     "consumer_protection"),

    ("How are consumer disputes referred to mediation?",
     "consumer_protection"),

    ("What is the purpose of consumer mediation?",
     "consumer_protection"),

    ("What happens when a consumer mediation settlement is reached?",
     "consumer_protection"),

    ("How does the Act address misleading advertisements?",
     "consumer_protection"),

    ("What penalties can apply to manufacturers of spurious goods?",
     "consumer_protection"),

    ("What penalties can apply for selling spurious goods?",
     "consumer_protection"),

    ("How does the Act address e-commerce transactions?",
     "consumer_protection"),

    ("What information must an e-commerce entity provide to consumers?",
     "consumer_protection"),

    ("What is the purpose of product liability provisions?",
     "consumer_protection"),

    ("When can a product manufacturer be liable for harm?",
     "consumer_protection"),

    ("When can a product seller be liable for harm?",
     "consumer_protection"),

    ("When can a product service provider be liable?",
     "consumer_protection"),

    ("What remedies may a consumer commission order?",
     "consumer_protection"),


    # ========================================================
    # OUT-OF-DOMAIN — 40
    # ========================================================

    ("What are the provisions of the Indian Contract Act concerning free consent?",
     "out_of_domain"),

    ("How is a company incorporated under the Companies Act?",
     "out_of_domain"),

    ("What are the requirements for filing an income tax return?",
     "out_of_domain"),

    ("How does GST registration work for a business?",
     "out_of_domain"),

    ("What is the procedure for obtaining a trademark?",
     "out_of_domain"),

    ("How does patent registration work in India?",
     "out_of_domain"),

    ("What does the Copyright Act protect?",
     "out_of_domain"),

    ("What are the provisions governing cheque dishonour?",
     "out_of_domain"),

    ("How does the Motor Vehicles Act regulate road transport?",
     "out_of_domain"),

    ("What are the requirements for obtaining a driving licence?",
     "out_of_domain"),

    ("How can an Indian citizen apply for a passport?",
     "out_of_domain"),

    ("What does the Information Technology Act say about cyber offences?",
     "out_of_domain"),

    ("What are the rules concerning data protection in India?",
     "out_of_domain"),

    ("What does the Arbitration and Conciliation Act provide?",
     "out_of_domain"),

    ("How does the Insolvency and Bankruptcy Code handle insolvency?",
     "out_of_domain"),

    ("What are the regulations governing Indian securities markets?",
     "out_of_domain"),

    ("What powers does SEBI have?",
     "out_of_domain"),

    ("What does the Right to Information Act provide?",
     "out_of_domain"),

    ("What does the Environmental Protection Act regulate?",
     "out_of_domain"),

    ("What are the provisions of the Air Act concerning pollution?",
     "out_of_domain"),

    ("How does the Water Act regulate water pollution?",
     "out_of_domain"),

    ("What does the Prevention of Corruption Act provide?",
     "out_of_domain"),

    ("What are the provisions of the Indian Penal Code concerning theft?",
     "out_of_domain"),

    ("How does criminal procedure work under Indian criminal law?",
     "out_of_domain"),

    ("What does the Evidence Act provide regarding admissibility of evidence?",
     "out_of_domain"),

    ("What are the provisions governing industrial disputes?",
     "out_of_domain"),

    ("How are minimum wages determined under Indian labour law?",
     "out_of_domain"),

    ("What does the Employees' Provident Funds law provide?",
     "out_of_domain"),

    ("What are the provisions concerning maternity benefits?",
     "out_of_domain"),

    ("How does the Patents Act protect pharmaceutical inventions?",
     "out_of_domain"),

    ("What are the rules governing foreign exchange transactions?",
     "out_of_domain"),

    ("What does FEMA regulate in India?",
     "out_of_domain"),

    ("What are the provisions of the Companies Act concerning directors?",
     "out_of_domain"),

    ("How does competition law regulate anti-competitive agreements?",
     "out_of_domain"),

    ("What does the Competition Act provide regarding abuse of dominance?",
     "out_of_domain"),

    ("How is a trademark infringement claim handled?",
     "out_of_domain"),

    ("What are the rules governing income tax deductions?",
     "out_of_domain"),

    ("How does customs law regulate imported goods?",
     "out_of_domain"),

    ("What does the National Food Security Act provide?",
     "out_of_domain"),

    ("What does the Right to Education Act provide?",
     "out_of_domain"),
]


def main():

    print("=" * 75)
    print("FINAL FROZEN BLIND EVALUATION")
    print("=" * 75)
    print(f"Total questions: {len(TEST_SET)}")
    print("Constitution: 30")
    print("Consumer Protection: 30")
    print("Out-of-domain: 40")
    print("=" * 75)

    engine = LegalRAGEngine()

    y_true = []
    y_pred = []
    rows = []

    for i, (question, expected) in enumerate(TEST_SET, 1):

        try:
            _, _, _, predicted = engine.retrieve(question)

            y_true.append(expected)
            y_pred.append(predicted)

            correct = expected == predicted

            rows.append([
                i,
                question,
                expected,
                predicted,
                correct
            ])

            print(
                f"[{i:03d}/100] "
                f"{'OK' if correct else 'ERROR':5s} | "
                f"Expected={expected:20s} | "
                f"Predicted={predicted:20s}"
            )

        except Exception as e:

            y_true.append(expected)
            y_pred.append("ERROR")

            rows.append([
                i,
                question,
                expected,
                "ERROR",
                False
            ])

            print(f"[{i:03d}/100] ERROR | {question}")
            print(f"          {e}")

    # ========================================================
    # 3-CLASS METRICS
    # ========================================================

    labels = [
        "constitution",
        "consumer_protection",
        "out_of_domain"
    ]

    accuracy_3 = accuracy_score(y_true, y_pred)

    p3, r3, f3, support3 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    cm3 = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # ========================================================
    # BINARY IN-DOMAIN / OUT-OF-DOMAIN METRICS
    # ========================================================

    y_true_binary = [
        "in_domain"
        if x in ("constitution", "consumer_protection")
        else "out_of_domain"
        for x in y_true
    ]

    y_pred_binary = [
        "in_domain"
        if x in ("constitution", "consumer_protection")
        else "out_of_domain"
        for x in y_pred
    ]

    accuracy_binary = accuracy_score(
        y_true_binary,
        y_pred_binary
    )

    pb, rb, fb, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        labels=["out_of_domain", "in_domain"],
        zero_division=0
    )

    cmb = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=["out_of_domain", "in_domain"]
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 75)
    print("FINAL 3-CLASS RESULTS")
    print("=" * 75)

    print(f"Accuracy: {accuracy_3 * 100:.4f}%")
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0,
            digits=4
        )
    )

    print("3-Class Confusion Matrix")
    print("Rows = Actual")
    print("Columns = Predicted")
    print()
    print(cm3)

    print("\n")
    print("=" * 75)
    print("FINAL BINARY GUARDRAIL RESULTS")
    print("=" * 75)

    print(f"Accuracy:  {accuracy_binary * 100:.4f}%")
    print(f"Precision: {pb[1] * 100:.4f}%")
    print(f"Recall:    {rb[1] * 100:.4f}%")
    print(f"F1-score:  {fb[1] * 100:.4f}%")

    print()
    print("Binary Confusion Matrix")
    print("Rows = Actual")
    print("Columns = Predicted")
    print()
    print(cmb)

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    errors = [
        row for row in rows
        if not row[4]
    ]

    print("\n")
    print("=" * 75)
    print(f"FINAL ERROR COUNT: {len(errors)} / {len(TEST_SET)}")
    print("=" * 75)

    for row in errors:
        print()
        print(f"Question #{row[0]}")
        print(f"Question : {row[1]}")
        print(f"Expected : {row[2]}")
        print(f"Predicted: {row[3]}")

    # ========================================================
    # SAVE EXACT RESULTS
    # ========================================================

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "FINAL_BLIND_EVALUATION_100.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "ID",
            "Question",
            "Expected",
            "Predicted",
            "Correct"
        ])

        writer.writerows(rows)

    report_path = output_dir / "FINAL_BLIND_EVALUATION_REPORT.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("FINAL FROZEN BLIND EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total questions: {len(TEST_SET)}\n")
        f.write("Constitution: 30\n")
        f.write("Consumer Protection: 30\n")
        f.write("Out-of-domain: 40\n\n")

        f.write("3-CLASS RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy: {accuracy_3 * 100:.4f}%\n\n")

        f.write("3-Class Confusion Matrix:\n")
        f.write(str(cm3))
        f.write("\n\n")

        f.write("BINARY RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy_binary * 100:.4f}%\n")
        f.write(f"Precision: {pb[1] * 100:.4f}%\n")
        f.write(f"Recall:    {rb[1] * 100:.4f}%\n")
        f.write(f"F1-score:  {fb[1] * 100:.4f}%\n\n")

        f.write("Binary Confusion Matrix:\n")
        f.write(str(cmb))
        f.write("\n\n")

        f.write(f"Errors: {len(errors)} / {len(TEST_SET)}\n\n")

        f.write("ERROR ANALYSIS\n")
        f.write("-" * 70 + "\n")

        for row in errors:
            f.write(f"\nQuestion #{row[0]}\n")
            f.write(f"Question: {row[1]}\n")
            f.write(f"Expected: {row[2]}\n")
            f.write(f"Predicted: {row[3]}\n")

    print("\n")
    print("=" * 75)
    print("RESULT FILES")
    print("=" * 75)
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
