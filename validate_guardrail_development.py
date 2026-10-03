from rag_engine import LegalRAGEngine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# NEW DEVELOPMENT VALIDATION SET
# Do not modify rag_engine.py based on these questions until
# the complete validation result has been recorded.

TESTS = [

    # -----------------------------
    # Constitution — 20
    # -----------------------------
    ("What does Article 14 guarantee?", "constitution"),
    ("What constitutional right is protected by Article 19?", "constitution"),
    ("What does Article 21 protect?", "constitution"),
    ("What is the purpose of Fundamental Duties?", "constitution"),
    ("What are the Directive Principles intended to achieve?", "constitution"),
    ("What is the constitutional status of the President?", "constitution"),
    ("How is the Vice-President elected under the Constitution?", "constitution"),
    ("What is the constitutional role of the Supreme Court?", "constitution"),
    ("What powers does the Election Commission have under the Constitution?", "constitution"),
    ("What does the Seventh Schedule contain?", "constitution"),
    ("What is the Concurrent List?", "constitution"),
    ("How are constitutional amendments made?", "constitution"),
    ("What is the constitutional position of local self-government?", "constitution"),
    ("What does Article 32 provide?", "constitution"),
    ("What does Article 226 empower High Courts to do?", "constitution"),
    ("What constitutional provision deals with free legal aid?", "constitution"),
    ("What does the Constitution provide about village panchayats?", "constitution"),
    ("What is the constitutional procedure for declaring an emergency?", "constitution"),
    ("What does the Constitution provide concerning the Finance Commission?", "constitution"),
    ("What is the constitutional position of the Comptroller and Auditor-General?", "constitution"),

    # -----------------------------
    # Consumer Protection — 20
    # -----------------------------
    ("Who qualifies as a consumer under the Consumer Protection Act?", "consumer_protection"),
    ("What is meant by defect in goods?", "consumer_protection"),
    ("What is deficiency in service?", "consumer_protection"),
    ("What is an unfair contract?", "consumer_protection"),
    ("What is product liability?", "consumer_protection"),
    ("What is an unfair trade practice?", "consumer_protection"),
    ("What is a restrictive trade practice?", "consumer_protection"),
    ("What is the role of the CCPA?", "consumer_protection"),
    ("How does the Act regulate misleading advertisements?", "consumer_protection"),
    ("What are the consumer dispute redressal commissions?", "consumer_protection"),
    ("What is the role of the District Consumer Commission?", "consumer_protection"),
    ("What is the role of the State Consumer Commission?", "consumer_protection"),
    ("What is the role of the National Consumer Commission?", "consumer_protection"),
    ("How does consumer mediation work?", "consumer_protection"),
    ("What happens after a consumer mediation settlement?", "consumer_protection"),
    ("How does the Act regulate online consumer transactions?", "consumer_protection"),
    ("What remedies can a consumer commission provide?", "consumer_protection"),
    ("Who can file a consumer complaint?", "consumer_protection"),
    ("What provisions deal with spurious goods?", "consumer_protection"),
    ("What is the purpose of product liability provisions?", "consumer_protection"),

    # -----------------------------
    # NEW OUT-OF-DOMAIN — 30
    # -----------------------------
    ("How is a partnership firm registered in India?", "out_of_domain"),
    ("What are the provisions concerning negotiable instruments?", "out_of_domain"),
    ("How does trademark opposition work?", "out_of_domain"),
    ("What is the procedure for patent examination?", "out_of_domain"),
    ("How does copyright infringement work?", "out_of_domain"),
    ("What are the rules for corporate mergers?", "out_of_domain"),
    ("How are company directors appointed?", "out_of_domain"),
    ("What is the process for corporate liquidation?", "out_of_domain"),
    ("How is GST input tax credit calculated?", "out_of_domain"),
    ("What are the income tax provisions for capital gains?", "out_of_domain"),
    ("How does customs valuation work?", "out_of_domain"),
    ("What are the rules governing foreign exchange?", "out_of_domain"),
    ("How does SEBI regulate mutual funds?", "out_of_domain"),
    ("What are the rules concerning insider trading?", "out_of_domain"),
    ("How are industrial disputes resolved?", "out_of_domain"),
    ("What are the rules governing employee provident funds?", "out_of_domain"),
    ("How are maternity benefits regulated?", "out_of_domain"),
    ("What does the Bharatiya Nyaya Sanhita provide regarding theft?", "out_of_domain"),
    ("How does criminal procedure operate in India?", "out_of_domain"),
    ("What are the rules concerning admissibility of criminal evidence?", "out_of_domain"),
    ("How does the Information Technology Act address cybercrime?", "out_of_domain"),
    ("What does data protection law regulate?", "out_of_domain"),
    ("How does the Air Act regulate air pollution?", "out_of_domain"),
    ("How does the Water Act regulate water pollution?", "out_of_domain"),
    ("What are the provisions of the Prevention of Corruption Act?", "out_of_domain"),
    ("How does the Arbitration and Conciliation Act work?", "out_of_domain"),
    ("What is the procedure for insolvency under the IBC?", "out_of_domain"),
    ("How does competition law address abuse of dominance?", "out_of_domain"),
    ("What does the Right to Information Act provide?", "out_of_domain"),
    ("How does the Motor Vehicles Act regulate driving licences?", "out_of_domain"),
]


def main():

    print("=" * 70)
    print("DEVELOPMENT VALIDATION — NEW QUESTIONS")
    print("=" * 70)
    print(f"Total: {len(TESTS)}")
    print("Constitution: 20")
    print("Consumer Protection: 20")
    print("Out-of-domain: 30")
    print("=" * 70)

    engine = LegalRAGEngine()

    y_true = []
    y_pred = []

    for i, (question, expected) in enumerate(TESTS, 1):

        try:
            _, _, _, predicted = engine.retrieve(question)

            y_true.append(expected)
            y_pred.append(predicted)

            status = "OK" if expected == predicted else "ERROR"

            print(
                f"[{i:02d}/{len(TESTS)}] {status:5s} | "
                f"Expected={expected:20s} | "
                f"Predicted={predicted:20s} | "
                f"{question}"
            )

        except Exception as e:
            y_true.append(expected)
            y_pred.append("ERROR")

            print(
                f"[{i:02d}/{len(TESTS)}] ERROR | "
                f"{question} | {e}"
            )

    labels = [
        "constitution",
        "consumer_protection",
        "out_of_domain"
    ]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print()
    print("=" * 70)
    print("DEVELOPMENT VALIDATION RESULTS")
    print("=" * 70)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print()

    for label, p, r, f, s in zip(
        labels,
        precision,
        recall,
        f1,
        support
    ):
        print(
            f"{label:22s} "
            f"Precision={p*100:.2f}% "
            f"Recall={r*100:.2f}% "
            f"F1={f*100:.2f}% "
            f"Support={s}"
        )

    print()
    print("Confusion Matrix:")
    print(cm)

    errors = [
        (i + 1, TESTS[i][0], TESTS[i][1], y_pred[i])
        for i in range(len(TESTS))
        if TESTS[i][1] != y_pred[i]
    ]

    print()
    print("=" * 70)
    print(f"ERRORS: {len(errors)} / {len(TESTS)}")
    print("=" * 70)

    for number, question, expected, predicted in errors:
        print()
        print(f"#{number}")
        print(f"Question : {question}")
        print(f"Expected : {expected}")
        print(f"Predicted: {predicted}")


if __name__ == "__main__":
    main()
