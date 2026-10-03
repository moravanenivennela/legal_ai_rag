"""
RAG Retrieval Evaluation Script (500 Questions)
Evaluates dense + sparse + RRF + cross-encoder reranking on legal questions.
Supports loading from outputs/rag_evaluation_questions_500.csv, incremental
saving/resuming, per-source breakdown, and Hit@1, Hit@3, Hit@5, MRR metrics.
"""

import os
import sys
import csv
import time
import json
import argparse
import datetime
from pathlib import Path
from rag_engine import LegalRAGEngine

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Preserved 30-question fallback list for backward compatibility
LEGACY_TESTS = [
    # Constitution
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

    # Consumer Protection
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


def load_dataset(dataset_path: str):
    """Load questions from CSV dataset or fall back to legacy 30 questions."""
    path = Path(dataset_path)
    if path.exists():
        questions = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                questions.append({
                    "question_id": row.get("question_id", f"Q{i:03d}"),
                    "question": row.get("question", "").strip(),
                    "expected_source": row.get("expected_source", "").strip(),
                    "question_type": row.get("question_type", "unspecified").strip(),
                    "source_reference": row.get("source_reference", "").strip(),
                    "reference_answer": row.get("reference_answer", "").strip(),
                })
        print(f"Loaded {len(questions)} questions from {dataset_path}")
        return questions
    else:
        print(f"Warning: {dataset_path} not found. Using legacy 30 questions.")
        questions = []
        for i, (q, src) in enumerate(LEGACY_TESTS, 1):
            prefix = "CON" if "constitution" in src else "CPA"
            questions.append({
                "question_id": f"LEGACY_{prefix}_{i:02d}",
                "question": q,
                "expected_source": src,
                "question_type": "factual",
                "source_reference": "",
                "reference_answer": "",
            })
        return questions


def get_source(hit):
    return hit.get("metadata", {}).get("source", "")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval on Legal Questions")
    parser.add_argument("--dataset", type=str, default="outputs/rag_evaluation_questions_500.csv",
                        help="Path to questions dataset CSV")
    parser.add_argument("--output-csv", type=str, default="outputs/RAG_RETRIEVAL_EVALUATION_500.csv",
                        help="Output path for results CSV")
    parser.add_argument("--output-report", type=str, default="outputs/RAG_RETRIEVAL_EVALUATION_500_REPORT.txt",
                        help="Output path for report text")
    parser.add_argument("--output-summary", type=str, default="outputs/RAG_RETRIEVAL_500_SUMMARY.json",
                        help="Output path for JSON summary metrics")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit evaluation to first N questions (for smoke testing)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing output CSV without re-evaluating completed questions")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh, overwriting existing output CSV")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique run ID (default: timestamp-based)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or datetime.datetime.now().strftime("RUN_%Y%m%d_%H%M%S")

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Run ID:        {run_id}")
    print(f"Dataset:       {args.dataset}")
    print(f"Output CSV:    {args.output_csv}")
    print(f"Output Report: {args.output_report}")
    print(f"Resume Mode:   {args.resume}")
    if args.limit:
        print(f"Limit:         {args.limit} questions (smoke test)")
    print("=" * 70)

    questions = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        questions = questions[:args.limit]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "question",
        "expected_source",
        "question_type",
        "rank",
        "hit1",
        "hit3",
        "hit5",
        "rr",
        "retrieval_success",
        "predicted_domain",
        "min_distance",
        "retrieved_sources",
        "latency_seconds",
        "status",
        "error",
        "run_id"
    ]

    # Resume handling: load already completed question_ids
    completed_ids = set()
    existing_rows = []
    if args.resume and output_csv.exists():
        try:
            with open(output_csv, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    qid = r.get("question_id")
                    if qid:
                        completed_ids.add(qid)
                        existing_rows.append(r)
            print(f"Resuming: found {len(completed_ids)} already completed questions.")
        except Exception as e:
            print(f"Warning reading existing CSV for resume: {e}")
            existing_rows = []
            completed_ids = set()

    # Filter out already evaluated questions (ensure each question evaluated exactly once)
    questions_to_evaluate = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Remaining questions to evaluate: {len(questions_to_evaluate)}/{len(questions)}")

    # Initialize CSV header if file does not exist or not resuming
    if not output_csv.exists() or not args.resume:
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    print("\nInitializing LegalRAGEngine...")
    engine = LegalRAGEngine()
    print("Engine initialized successfully.\n")

    results = list(existing_rows)

    total_to_run = len(questions)
    for idx, q_item in enumerate(questions, 1):
        qid = q_item["question_id"]
        question = q_item["question"]
        expected_source = q_item["expected_source"]
        q_type = q_item.get("question_type", "")

        # Skip if already completed
        if qid in completed_ids:
            continue

        start_time = time.perf_counter()
        error_msg = ""
        rank = None
        hit1 = False
        hit3 = False
        hit5 = False
        rr = 0.0
        retrieval_success = False
        predicted_domain = ""
        min_distance = 999.0
        retrieved_sources_str = ""

        try:
            hits, retrieval_success, min_distance, predicted_domain = engine.retrieve(question)
            sources = [get_source(hit) for hit in hits]
            retrieved_sources_str = ";".join(sources)

            for position, source in enumerate(sources, 1):
                if source == expected_source:
                    rank = position
                    break

            hit1 = rank is not None and rank <= 1
            hit3 = rank is not None and rank <= 3
            hit5 = rank is not None and rank <= 5
            rr = 1.0 / rank if rank is not None else 0.0
            status = "OK" if rank is not None else "MISS"

        except Exception as e:
            error_msg = str(e)
            status = "ERROR"

        latency = round(time.perf_counter() - start_time, 4)

        row = {
            "question_id": qid,
            "question": question,
            "expected_source": expected_source,
            "question_type": q_type,
            "rank": rank if rank is not None else "",
            "hit1": hit1,
            "hit3": hit3,
            "hit5": hit5,
            "rr": round(rr, 4),
            "retrieval_success": retrieval_success,
            "predicted_domain": predicted_domain,
            "min_distance": round(min_distance, 4) if isinstance(min_distance, (int, float)) else min_distance,
            "retrieved_sources": retrieved_sources_str,
            "latency_seconds": latency,
            "status": status,
            "error": error_msg,
            "run_id": run_id
        }

        results.append(row)
        completed_ids.add(qid)

        # Incremental write: flush immediately so work is never lost
        with open(output_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        rank_display = f"Rank={rank}" if rank is not None else "Rank=None"
        pct = (len(completed_ids) / total_to_run) * 100
        print(
            f"[{len(completed_ids):03d}/{total_to_run}] ({pct:5.1f}%) {status:5s} | "
            f"{rank_display:8s} | "
            f"Domain={predicted_domain:19s} | "
            f"Latency={latency:.3f}s | "
            f"{question[:55]}"
        )

    # Calculate actual metrics from current results
    total_evaluated = len(results)
    if total_evaluated == 0:
        print("\nNo evaluations completed.")
        return

    def parse_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")

    def parse_float(v, default=0.0):
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    valid_results = [r for r in results if r.get("status") != "ERROR"]
    error_results = [r for r in results if r.get("status") == "ERROR"]

    hit1_count = sum(1 for r in valid_results if parse_bool(r.get("hit1")))
    hit3_count = sum(1 for r in valid_results if parse_bool(r.get("hit3")))
    hit5_count = sum(1 for r in valid_results if parse_bool(r.get("hit5")))
    mrr_sum = sum(parse_float(r.get("rr")) for r in valid_results)
    retrieval_success_count = sum(1 for r in valid_results if parse_bool(r.get("retrieval_success")))

    hit1_pct = (hit1_count / total_evaluated) * 100
    hit3_pct = (hit3_count / total_evaluated) * 100
    hit5_pct = (hit5_count / total_evaluated) * 100
    mrr = mrr_sum / total_evaluated
    retrieval_success_pct = (retrieval_success_count / total_evaluated) * 100

    latencies = [parse_float(r.get("latency_seconds")) for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_latencies = sorted(latencies)
    median_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0

    # Per-source breakdown
    const_results = [r for r in results if "constitution" in r.get("expected_source", "").lower()]
    cp_results = [r for r in results if "consumer" in r.get("expected_source", "").lower()]

    def calc_source_metrics(src_rows):
        n = len(src_rows)
        if n == 0:
            return {"total": 0, "hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr": 0.0, "retrieval_success": 0.0}
        h1 = sum(1 for r in src_rows if parse_bool(r.get("hit1"))) / n * 100
        h3 = sum(1 for r in src_rows if parse_bool(r.get("hit3"))) / n * 100
        h5 = sum(1 for r in src_rows if parse_bool(r.get("hit5"))) / n * 100
        src_mrr = sum(parse_float(r.get("rr")) for r in src_rows) / n
        succ = sum(1 for r in src_rows if parse_bool(r.get("retrieval_success"))) / n * 100
        return {"total": n, "hit1": h1, "hit3": h3, "hit5": h5, "mrr": src_mrr, "retrieval_success": succ}

    const_metrics = calc_source_metrics(const_results)
    cp_metrics = calc_source_metrics(cp_results)

    misses = [r for r in results if not parse_bool(r.get("hit5"))]

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL RESULTS (500-QUESTION EVALUATION)")
    print("=" * 70)
    print(f"Total Evaluated:        {total_evaluated}")
    print(f"Retrieval Success:      {retrieval_success_count}/{total_evaluated} ({retrieval_success_pct:.2f}%)")
    print(f"Hit@1:                  {hit1_pct:.2f}% ({hit1_count}/{total_evaluated})")
    print(f"Hit@3:                  {hit3_pct:.2f}% ({hit3_count}/{total_evaluated})")
    print(f"Hit@5:                  {hit5_pct:.2f}% ({hit5_count}/{total_evaluated})")
    print(f"MRR:                    {mrr:.4f}")
    print(f"Average Latency:        {avg_latency:.4f}s")
    print(f"Median Latency:         {median_latency:.4f}s")
    print(f"Errors/Exceptions:      {len(error_results)}")
    print(f"Total Misses (not in Top-5): {len(misses)}")
    print("-" * 70)
    print(f"Constitution of India:  Hit@1={const_metrics['hit1']:.2f}% | Hit@3={const_metrics['hit3']:.2f}% | Hit@5={const_metrics['hit5']:.2f}% | MRR={const_metrics['mrr']:.4f} (N={const_metrics['total']})")
    print(f"Consumer Protection:    Hit@1={cp_metrics['hit1']:.2f}% | Hit@3={cp_metrics['hit3']:.2f}% | Hit@5={cp_metrics['hit5']:.2f}% | MRR={cp_metrics['mrr']:.4f} (N={cp_metrics['total']})")
    print("=" * 70)

    # Save detailed text report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RAG RETRIEVAL EVALUATION REPORT (500 QUESTIONS)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Run ID:                 {run_id}\n")
        f.write(f"Timestamp:              {datetime.datetime.now().isoformat()}\n")
        f.write(f"Dataset:                {args.dataset}\n")
        f.write(f"Total Evaluated:        {total_evaluated}\n")
        f.write(f"Retrieval Success:      {retrieval_success_count}/{total_evaluated} ({retrieval_success_pct:.2f}%)\n")
        f.write(f"Hit@1:                  {hit1_pct:.2f}%\n")
        f.write(f"Hit@3:                  {hit3_pct:.2f}%\n")
        f.write(f"Hit@5:                  {hit5_pct:.2f}%\n")
        f.write(f"MRR:                    {mrr:.4f}\n")
        f.write(f"Average Latency:        {avg_latency:.4f}s\n")
        f.write(f"Median Latency:         {median_latency:.4f}s\n")
        f.write(f"Errors:                 {len(error_results)}\n\n")

        f.write("PER-SOURCE BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"Constitution of India (N={const_metrics['total']}):\n")
        f.write(f"  Retrieval Success:    {const_metrics['retrieval_success']:.2f}%\n")
        f.write(f"  Hit@1:                {const_metrics['hit1']:.2f}%\n")
        f.write(f"  Hit@3:                {const_metrics['hit3']:.2f}%\n")
        f.write(f"  Hit@5:                {const_metrics['hit5']:.2f}%\n")
        f.write(f"  MRR:                  {const_metrics['mrr']:.4f}\n\n")

        f.write(f"Consumer Protection Act 2019 (N={cp_metrics['total']}):\n")
        f.write(f"  Retrieval Success:    {cp_metrics['retrieval_success']:.2f}%\n")
        f.write(f"  Hit@1:                {cp_metrics['hit1']:.2f}%\n")
        f.write(f"  Hit@3:                {cp_metrics['hit3']:.2f}%\n")
        f.write(f"  Hit@5:                {cp_metrics['hit5']:.2f}%\n")
        f.write(f"  MRR:                  {cp_metrics['mrr']:.4f}\n\n")

        f.write("NOTE ON EVALUATION METHODOLOGY\n")
        f.write("-" * 70 + "\n")
        f.write("Source matching and retrieval success verify that the retrieval pipeline returned\n")
        f.write("contexts originating from the ground-truth legal document. They do NOT by themselves\n")
        f.write("prove substantive legal accuracy of downstream answers.\n\n")

        if misses:
            f.write(f"RETRIEVAL MISSES (Top-5 Misses: {len(misses)})\n")
            f.write("-" * 70 + "\n")
            for m in misses:
                f.write(f"[{m.get('question_id')}] Rank={m.get('rank')} | Expected={m.get('expected_source')}\n")
                f.write(f"  Question:  {m.get('question')}\n")
                f.write(f"  Retrieved: {m.get('retrieved_sources')}\n\n")

    # Save summary JSON
    summary_data = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": args.dataset,
        "total_evaluated": total_evaluated,
        "retrieval_success_rate": round(retrieval_success_pct, 2),
        "hit1_percent": round(hit1_pct, 2),
        "hit3_percent": round(hit3_pct, 2),
        "hit5_percent": round(hit5_pct, 2),
        "mrr": round(mrr, 4),
        "average_latency_seconds": round(avg_latency, 4),
        "median_latency_seconds": round(median_latency, 4),
        "errors_count": len(error_results),
        "misses_count": len(misses),
        "constitution_metrics": const_metrics,
        "consumer_protection_metrics": cp_metrics
    }

    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\nSaved files:")
    print(f"  CSV:     {output_csv}")
    print(f"  Report:  {report_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
