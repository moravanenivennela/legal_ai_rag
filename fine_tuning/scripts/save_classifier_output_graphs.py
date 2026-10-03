import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------
DATA_DIR = Path("fine_tuning/dataset/expanded/classification/repaired_v3")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_MODEL = Path("models/query_classifier_v3_candidate.pkl")
EXISTING_MODEL = Path("query_classifier.pkl")

TRAIN_FILE = DATA_DIR / "train_classification.jsonl"
VALIDATION_FILE = DATA_DIR / "validation_classification.jsonl"
TEST_FILE = DATA_DIR / "test_classification.jsonl"

LABELS = ["constitution", "consumer_protection", "out_of_domain"]


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def save_confusion_matrix(y_true, y_pred, title, output_path):
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return matrix.tolist()


def evaluate(model, rows, vectors):
    y_true = [row["label"] for row in rows]
    y_pred = model.predict(vectors).tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "sample_count": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }

    for i, label in enumerate(LABELS):
        metrics["per_class"][label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    errors = []
    for row, prediction in zip(rows, y_pred):
        if row["label"] != prediction:
            errors.append({
                "expected_label": row["label"],
                "predicted_label": prediction,
                "question": row["text"],
            })

    return metrics, y_true, y_pred, errors


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    required = [
        CANDIDATE_MODEL, EXISTING_MODEL,
        TRAIN_FILE, VALIDATION_FILE, TEST_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files not found:\n" + "\n".join(missing)
        )

    print("Loading datasets and classifiers...")
    train_rows = load_jsonl(TRAIN_FILE)
    val_rows = load_jsonl(VALIDATION_FILE)
    test_rows = load_jsonl(TEST_FILE)

    candidate = load_pickle(CANDIDATE_MODEL)
    existing = load_pickle(EXISTING_MODEL)

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Embed each split once, then use the same vectors for both models.
    train_vectors = np.asarray(
        embedder.embed_documents([row["text"] for row in train_rows])
    )
    val_vectors = np.asarray(
        embedder.embed_documents([row["text"] for row in val_rows])
    )
    test_vectors = np.asarray(
        embedder.embed_documents([row["text"] for row in test_rows])
    )

    print("Evaluating candidate classifier...")
    candidate_val, val_true, val_pred, val_errors = evaluate(
        candidate, val_rows, val_vectors
    )
    candidate_test, test_true, candidate_test_pred, candidate_errors = evaluate(
        candidate, test_rows, test_vectors
    )

    print("Evaluating existing classifier on the same test set...")
    existing_test, _, existing_test_pred, existing_errors = evaluate(
        existing, test_rows, test_vectors
    )

    # Confusion matrices
    save_confusion_matrix(
        val_true, val_pred,
        "Candidate Classifier - Validation Confusion Matrix",
        OUTPUT_DIR / "classifier_v3_validation_confusion_matrix.png",
    )
    save_confusion_matrix(
        test_true, candidate_test_pred,
        "Candidate Classifier - Test Confusion Matrix",
        OUTPUT_DIR / "classifier_v3_candidate_test_confusion_matrix.png",
    )
    save_confusion_matrix(
        test_true, existing_test_pred,
        "Existing Classifier - Same Test Set Confusion Matrix",
        OUTPUT_DIR / "classifier_existing_test_confusion_matrix.png",
    )

    # Compare overall metrics
    model_names = ["Existing classifier", "V3 candidate"]
    model_metrics = [existing_test, candidate_test]
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]

    x = np.arange(len(metric_keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        [m[k] * 100 for m in model_metrics[:1]] * 1
        if False else [existing_test[k] * 100 for k in metric_keys],
        width, label=model_names[0],
    )
    ax.bar(
        x + width / 2,
        [candidate_test[k] * 100 for k in metric_keys],
        width, label=model_names[1],
    )
    ax.set_ylabel("Score (%)")
    ax.set_title("Existing vs V3 Candidate Classifier - Test Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "classifier_v3_model_comparison.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    # Per-class precision, recall, and F1 for candidate test
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(LABELS))
    width = 0.24
    for offset, metric_key, label in [
        (-width, "precision", "Precision"),
        (0, "recall", "Recall"),
        (width, "f1", "F1-score"),
    ]:
        values = [
            candidate_test["per_class"][name][metric_key] * 100
            for name in LABELS
        ]
        ax.bar(x + offset, values, width, label=label)

    ax.set_ylabel("Score (%)")
    ax.set_title("V3 Candidate - Per-Class Test Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "classifier_v3_per_class_metrics.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    # Dataset class distribution
    split_data = {
        "Train": train_rows,
        "Validation": val_rows,
        "Test": test_rows,
    }
    counts = {
        split: [sum(1 for row in rows if row["label"] == label) for label in LABELS]
        for split, rows in split_data.items()
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(split_data))
    width = 0.24
    for i, label in enumerate(LABELS):
        ax.bar(
            x + (i - 1) * width,
            [counts[split][i] for split in split_data],
            width,
            label=label,
        )
    ax.set_ylabel("Number of questions")
    ax.set_title("V3 Classification Dataset Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(list(split_data.keys()))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "classifier_v3_dataset_distribution.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)

    # Save metric report and errors
    report = {
        "dataset_note": (
            "V3 includes synthetic OOD training candidates that have not all "
            "been independently human-verified. Results are preliminary."
        ),
        "label_order": LABELS,
        "dataset_counts": {
            split: {
                label: sum(1 for row in rows if row["label"] == label)
                for label in LABELS
            }
            for split, rows in split_data.items()
        },
        "candidate_validation": candidate_val,
        "candidate_test": candidate_test,
        "existing_classifier_same_test": existing_test,
    }
    save_json(OUTPUT_DIR / "classifier_v3_full_metrics_report.json", report)

    with (OUTPUT_DIR / "classifier_v3_misclassified_questions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "expected_label", "predicted_label", "question"],
        )
        writer.writeheader()
        for split_name, error_rows in [
            ("validation_candidate", val_errors),
            ("test_candidate", candidate_errors),
            ("test_existing_classifier", existing_errors),
        ]:
            for row in error_rows:
                writer.writerow({"split": split_name, **row})

    print("\n=== SAVED OUTPUTS ===")
    for path in sorted(OUTPUT_DIR.glob("classifier_v3_*")):
        print(path)
    print(OUTPUT_DIR / "classifier_existing_test_confusion_matrix.png")
    print("\nCandidate validation accuracy:", round(candidate_val["accuracy"] * 100, 2), "%")
    print("Candidate test accuracy:", round(candidate_test["accuracy"] * 100, 2), "%")
    print("Existing test accuracy:", round(existing_test["accuracy"] * 100, 2), "%")
    print("\nAll output graphs and reports saved successfully.")


if __name__ == "__main__":
    main()
