import json
import pickle
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

DATA_DIR = Path("fine_tuning/dataset/expanded/classification/repaired_v3")
MODEL_OUT = Path("models/query_classifier_v3_candidate.pkl")
REPORT_OUT = Path("fine_tuning/classifier_v3_candidate_report.json")
CM_OUT = Path("outputs/classifier_v3_candidate_confusion_matrix.png")

TRAIN_FILE = DATA_DIR / "train_classification.jsonl"
VAL_FILE = DATA_DIR / "validation_classification.jsonl"
TEST_FILE = DATA_DIR / "test_classification.jsonl"


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            if not text or label not in {
                "constitution", "consumer_protection", "out_of_domain"
            }:
                raise ValueError(
                    f"Invalid text/label in {path}, line {line_number}"
                )
            records.append({"text": text, "label": label})
    return records


def evaluate_model(name, model, X, y_true, labels):
    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }
    print(f"\n{name}")
    print("-" * 55)
    print(f"Accuracy:         {result['accuracy'] * 100:.2f}%")
    print(f"Macro precision:  {precision * 100:.2f}%")
    print(f"Macro recall:     {recall * 100:.2f}%")
    print(f"Macro F1:         {f1 * 100:.2f}%")
    print(classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    ))
    return result, y_pred


def main():
    train = load_jsonl(TRAIN_FILE)
    val = load_jsonl(VAL_FILE)
    test = load_jsonl(TEST_FILE)

    labels = ["constitution", "consumer_protection", "out_of_domain"]

    print("DATASET SIZES")
    print(f"Training:   {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")
    print("\nTraining class counts:")
    for label in labels:
        print(f"  {label}: {sum(r['label'] == label for r in train)}")

    print("\nLoading BGE-M3 embeddings...")
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    X_train = np.asarray(
        embedder.embed_documents([r["text"] for r in train])
    )
    X_val = np.asarray(
        embedder.embed_documents([r["text"] for r in val])
    )
    X_test = np.asarray(
        embedder.embed_documents([r["text"] for r in test])
    )
    y_train = np.array([r["label"] for r in train])
    y_val = np.array([r["label"] for r in val])
    y_test = np.array([r["label"] for r in test])

    print("\nTraining candidate Logistic Regression classifier...")
    candidate = LogisticRegression(max_iter=2000, C=1.0, class_weight=None)
    candidate.fit(X_train, y_train)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open("wb") as f:
        pickle.dump(candidate, f)
    print(f"Candidate model saved separately: {MODEL_OUT}")

    results = {
        "dataset": {
            "train_size": len(train),
            "validation_size": len(val),
            "test_size": len(test),
            "training_class_counts": {
                label: int(sum(y_train == label)) for label in labels
            },
            "test_class_counts": {
                label: int(sum(y_test == label)) for label in labels
            },
            "note": (
                "V3 includes synthetic OOD training candidates that have "
                "not all been human-verified. Metrics are preliminary."
            ),
        },
        "validation": {},
        "test": {},
    }

    val_result, _ = evaluate_model(
        "V3 candidate — validation split", candidate, X_val, y_val, labels
    )
    results["validation"] = val_result

    candidate_test_result, candidate_predictions = evaluate_model(
        "V3 candidate — test split", candidate, X_test, y_test, labels
    )
    results["test"]["v3_candidate"] = candidate_test_result

    # Compare the current classifier without replacing it.
    old_model_path = Path("query_classifier.pkl")
    if old_model_path.exists():
        try:
            with old_model_path.open("rb") as f:
                old_model = pickle.load(f)
            old_result, _ = evaluate_model(
                "Existing classifier — same test split",
                old_model, X_test, y_test, labels
            )
            results["test"]["existing_classifier"] = old_result
        except Exception as exc:
            print(f"\nCould not evaluate existing classifier: {exc}")
            results["test"]["existing_classifier_error"] = str(exc)
    else:
        print("\nExisting query_classifier.pkl not found; comparison skipped.")

    CM_OUT.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, candidate_predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("V3 Candidate Classifier — Test Confusion Matrix")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(CM_OUT, dpi=160)
    plt.close(fig)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFinished.")
    print(f"Report: {REPORT_OUT}")
    print(f"Confusion matrix: {CM_OUT}")
    print("The production query_classifier.pkl was NOT changed.")


if __name__ == "__main__":
    main()
