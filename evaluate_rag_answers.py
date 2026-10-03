"""
RAG Answer & Groundedness Evaluation Script (500 Questions)
Evaluates retrieval + LLM generation + DeBERTa NLI groundedness on legal questions.
Loads questions from outputs/rag_evaluation_questions_500.csv, supports incremental
checkpoints/resuming, per-source breakdown, and comprehensive metrics export.
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


def collect_answer(engine, question, contexts):
    """Stream answer from engine and join chunks."""
    chunks = []
    for token in engine.generate_stream(
        question,
        contexts,
        language="English",
        eli5=False
    ):
        chunks.append(token)
    return "".join(chunks).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG Answer Generation on Legal Questions")
    parser.add_argument("--dataset", type=str, default="outputs/rag_evaluation_questions_500.csv",
                        help="Path to questions dataset CSV")
    parser.add_argument("--output-csv", type=str, default="outputs/RAG_ANSWER_EVALUATION_500.csv",
                        help="Output path for results CSV")
    parser.add_argument("--output-report", type=str, default="outputs/RAG_ANSWER_EVALUATION_500_REPORT.txt",
                        help="Output path for report text")
    parser.add_argument("--output-summary", type=str, default="outputs/RAG_EVALUATION_500_SUMMARY.json",
                        help="Output path for combined JSON summary metrics")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit evaluation to first N questions (for smoke testing)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing output CSV without re-evaluating completed questions")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh, overwriting existing output CSV")
    parser.add_argument("--model", type=str, default="legal-ai-finetuned:latest",
                        help="Ollama model name to evaluate")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique run ID (default: timestamp-based)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or datetime.datetime.now().strftime("RUN_%Y%m%d_%H%M%S")

    print("=" * 70)
    print("RAG ANSWER / GROUNDEDNESS EVALUATION (500 QUESTIONS)")
    print("=" * 70)
    print(f"Run ID:        {run_id}")
    print(f"Model:         {args.model}")
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
        "source_reference",
        "predicted_domain",
        "retrieval_success",
        "source_match",
        "min_distance",
        "groundedness",
        "qa_f1",
        "qa_precision",
        "qa_recall",
        "latency_seconds",
        "status",
        "error",
        "run_id",
        "model_name",
        "answer",
        "reference_answer"
    ]

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

    questions_to_evaluate = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Remaining questions to evaluate: {len(questions_to_evaluate)}/{len(questions)}")

    if not output_csv.exists() or not args.resume:
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    print("\nInitializing LegalRAGEngine with model:", args.model)
    engine = LegalRAGEngine(model_name=args.model)
    print("Engine initialized successfully.\n")

    results = list(existing_rows)
    total_to_run = len(questions)

    for idx, q_item in enumerate(questions, 1):
        qid = q_item["question_id"]
        question = q_item["question"]
        expected_source = q_item["expected_source"]
        q_type = q_item.get("question_type", "")
        ref = q_item.get("source_reference", "")
        ref_ans = q_item.get("reference_answer", "")

        if qid in completed_ids:
            continue

        start_time = time.perf_counter()
        error_msg = ""
        predicted_domain = ""
        retrieval_success = False
        source_match = False
        min_distance = 999.0
        groundedness = 0.0
        qa_f1 = 0.0
        qa_prec = 0.0
        qa_rec = 0.0
        answer = ""
        status = "OK"

        try:
            contexts, retrieval_success, min_distance, predicted_domain = engine.retrieve(question)

            if not retrieval_success or not contexts:
                status = "RETRIEVAL_FAIL"
                error_msg = "Retrieval failed or returned empty contexts"
            else:
                sources = [c.get("metadata", {}).get("source", "") for c in contexts]
                source_match = expected_source in sources

                answer = collect_answer(engine, question, contexts)

                groundedness = engine.check_groundedness(answer, contexts)

                if ref_ans:
                    qa_metrics = engine.compute_qa_metrics(answer, ref_ans)
                    qa_f1 = qa_metrics.get("f1", 0.0)
                    qa_prec = qa_metrics.get("precision", 0.0)
                    qa_rec = qa_metrics.get("recall", 0.0)

        except Exception as e:
            error_msg = str(e)
            status = "ERROR"

        latency = round(time.perf_counter() - start_time, 2)

        row = {
            "question_id": qid,
            "question": question,
            "expected_source": expected_source,
            "question_type": q_type,
            "source_reference": ref,
            "predicted_domain": predicted_domain,
            "retrieval_success": retrieval_success,
            "source_match": source_match,
            "min_distance": round(min_distance, 4) if isinstance(min_distance, (int, float)) else min_distance,
            "groundedness": groundedness,
            "qa_f1": qa_f1,
            "qa_precision": qa_prec,
            "qa_recall": qa_rec,
            "latency_seconds": latency,
            "status": status,
            "error": error_msg,
            "run_id": run_id,
            "model_name": args.model,
            "answer": answer,
            "reference_answer": ref_ans
        }

        results.append(row)
        completed_ids.add(qid)

        # Incremental write
        with open(output_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        pct = (len(completed_ids) / total_to_run) * 100
        print(
            f"[{len(completed_ids):03d}/{total_to_run}] ({pct:5.1f}%) {status:14s} | "
            f"Match={str(source_match):5s} | "
            f"Grounded={groundedness:5.1f}% | "
            f"Latency={latency:5.2f}s | "
            f"{question[:45]}"
        )

    # Compute aggregate metrics
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

    successful_answers = [r for r in results if parse_bool(r.get("retrieval_success")) and r.get("answer")]
    error_count = sum(1 for r in results if r.get("status") == "ERROR")
    failed_retrieval_count = sum(1 for r in results if r.get("status") == "RETRIEVAL_FAIL")

    succ_n = len(successful_answers)
    source_match_pct = (sum(1 for r in successful_answers if parse_bool(r.get("source_match"))) / succ_n * 100) if succ_n else 0.0

    groundedness_scores = [parse_float(r.get("groundedness")) for r in successful_answers]
    avg_groundedness = sum(groundedness_scores) / succ_n if succ_n else 0.0
    sorted_groundedness = sorted(groundedness_scores)
    median_groundedness = sorted_groundedness[succ_n // 2] if succ_n else 0.0
    grounded_70_pct = (sum(1 for g in groundedness_scores if g >= 70.0) / succ_n * 100) if succ_n else 0.0

    latencies = [parse_float(r.get("latency_seconds")) for r in results]
    avg_latency = sum(latencies) / total_evaluated if total_evaluated else 0.0
    sorted_latencies = sorted(latencies)
    median_latency = sorted_latencies[total_evaluated // 2] if total_evaluated else 0.0

    # Per-source breakdown
    const_rows = [r for r in successful_answers if "constitution" in r.get("expected_source", "").lower()]
    cp_rows = [r for r in successful_answers if "consumer" in r.get("expected_source", "").lower()]

    def calc_source_answer_metrics(src_rows):
        n = len(src_rows)
        if n == 0:
            return {"count": 0, "source_match": 0.0, "avg_groundedness": 0.0, "grounded_70_pct": 0.0, "avg_latency": 0.0}
        sm = sum(1 for r in src_rows if parse_bool(r.get("source_match"))) / n * 100
        gs = [parse_float(r.get("groundedness")) for r in src_rows]
        ag = sum(gs) / n
        g70 = sum(1 for g in gs if g >= 70.0) / n * 100
        lats = [parse_float(r.get("latency_seconds")) for r in src_rows]
        al = sum(lats) / n
        return {
            "count": n,
            "source_match": round(sm, 2),
            "avg_groundedness": round(ag, 2),
            "grounded_70_pct": round(g70, 2),
            "avg_latency": round(al, 2)
        }

    const_answer_metrics = calc_source_answer_metrics(const_rows)
    cp_answer_metrics = calc_source_answer_metrics(cp_rows)

    print("\n" + "=" * 70)
    print("FINAL ANSWER EVALUATION RESULTS (500 QUESTIONS)")
    print("=" * 70)
    print(f"Total Evaluated:             {total_evaluated}")
    print(f"Successful Answers:          {succ_n}/{total_evaluated} ({succ_n/total_evaluated*100:.2f}%)")
    print(f"Expected Source Match:       {source_match_pct:.2f}%")
    print(f"Average Groundedness:        {avg_groundedness:.2f}%")
    print(f"Median Groundedness:         {median_groundedness:.2f}%")
    print(f"Groundedness >= 70%:         {grounded_70_pct:.2f}%")
    print(f"Average Latency:             {avg_latency:.2f}s")
    print(f"Median Latency:              {median_latency:.2f}s")
    print(f"Failed Retrievals:           {failed_retrieval_count}")
    print(f"Execution Errors:            {error_count}")
    print("-" * 70)
    print(f"Constitution (N={const_answer_metrics['count']}):   Match={const_answer_metrics['source_match']}% | Grounded={const_answer_metrics['avg_groundedness']}% | >=70%={const_answer_metrics['grounded_70_pct']}%")
    print(f"Consumer Prot (N={cp_answer_metrics['count']}): Match={cp_answer_metrics['source_match']}% | Grounded={cp_answer_metrics['avg_groundedness']}% | >=70%={cp_answer_metrics['grounded_70_pct']}%")
    print("=" * 70)

    # Save detailed text report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RAG ANSWER / GROUNDEDNESS EVALUATION REPORT (500 QUESTIONS)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Run ID:                      {run_id}\n")
        f.write(f"Timestamp:                   {datetime.datetime.now().isoformat()}\n")
        f.write(f"Model:                       {args.model}\n")
        f.write(f"Dataset:                     {args.dataset}\n")
        f.write(f"Total Evaluated:             {total_evaluated}\n")
        f.write(f"Successful Answers:          {succ_n}/{total_evaluated} ({succ_n/total_evaluated*100:.2f}%)\n")
        f.write(f"Expected Source Match:       {source_match_pct:.2f}%\n")
        f.write(f"Average Groundedness:        {avg_groundedness:.2f}%\n")
        f.write(f"Median Groundedness:         {median_groundedness:.2f}%\n")
        f.write(f"Groundedness >= 70%:         {grounded_70_pct:.2f}%\n")
        f.write(f"Average Latency:             {avg_latency:.2f}s\n")
        f.write(f"Median Latency:              {median_latency:.2f}s\n")
        f.write(f"Failed Retrievals:           {failed_retrieval_count}\n")
        f.write(f"Execution Errors:            {error_count}\n\n")

        f.write("METHODOLOGICAL NOTE ON METRICS INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write("Expected-source match, retrieval success, and NLI entailment scores are automated\n")
        f.write("groundedness metrics reflecting retrieved context alignment. They should NOT be described\n")
        f.write("as human-verified legal correctness. Reference answers and SQuAD token overlap provide\n")
        f.write("lexical reference metrics where gold reference answers are available.\n\n")

        f.write("PER-SOURCE BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"Constitution of India (N={const_answer_metrics['count']}):\n")
        f.write(f"  Source Match:              {const_answer_metrics['source_match']}%\n")
        f.write(f"  Average Groundedness:      {const_answer_metrics['avg_groundedness']}%\n")
        f.write(f"  Groundedness >= 70%:       {const_answer_metrics['grounded_70_pct']}%\n")
        f.write(f"  Average Latency:           {const_answer_metrics['avg_latency']}s\n\n")

        f.write(f"Consumer Protection Act 2019 (N={cp_answer_metrics['count']}):\n")
        f.write(f"  Source Match:              {cp_answer_metrics['source_match']}%\n")
        f.write(f"  Average Groundedness:      {cp_answer_metrics['avg_groundedness']}%\n")
        f.write(f"  Groundedness >= 70%:       {cp_answer_metrics['grounded_70_pct']}%\n")
        f.write(f"  Average Latency:           {cp_answer_metrics['avg_latency']}s\n\n")

        f.write("SAMPLE QUESTION RESULTS (First 15)\n")
        f.write("-" * 70 + "\n")
        for r in results[:15]:
            f.write(f"\nQuestion [{r.get('question_id')}]: {r.get('question')}\n")
            f.write(f"  Expected Source: {r.get('expected_source')}\n")
            f.write(f"  Status:          {r.get('status')}\n")
            f.write(f"  Source Match:    {r.get('source_match')}\n")
            f.write(f"  Groundedness:    {r.get('groundedness')}%\n")
            f.write(f"  Latency:         {r.get('latency_seconds')}s\n")
            ans_snippet = (r.get('answer') or '')[:200].replace('\n', ' ')
            f.write(f"  Answer:          {ans_snippet}...\n")

    # Update combined summary JSON
    summary_path = Path(args.output_summary)
    summary_data = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            summary_data = {}

    summary_data.update({
        "answer_evaluation": {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "model": args.model,
            "dataset": args.dataset,
            "total_evaluated": total_evaluated,
            "successful_answers": succ_n,
            "success_rate_percent": round(succ_n / total_evaluated * 100, 2) if total_evaluated else 0.0,
            "source_match_percent": round(source_match_pct, 2),
            "average_groundedness_percent": round(avg_groundedness, 2),
            "median_groundedness_percent": round(median_groundedness, 2),
            "groundedness_above_70_percent": round(grounded_70_pct, 2),
            "average_latency_seconds": round(avg_latency, 2),
            "median_latency_seconds": round(median_latency, 2),
            "failed_retrieval_count": failed_retrieval_count,
            "error_count": error_count,
            "constitution_metrics": const_answer_metrics,
            "consumer_protection_metrics": cp_answer_metrics
        }
    })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\nSaved files:")
    print(f"  CSV:     {output_csv}")
    print(f"  Report:  {report_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
