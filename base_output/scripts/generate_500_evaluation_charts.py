"""
Generate Evaluation Figures and Tables for 500-Question Legal RAG Benchmark.

Produces publication-quality charts in base_output/figures/
and structured CSV tables in base_output/tables/.
Compares baseline (N=30) with the full benchmark (N=500).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
BASE_OUTPUT = ROOT / "base_output"
FIGDIR = BASE_OUTPUT / "figures"
TABDIR = BASE_OUTPUT / "tables"
OUTPUTS = ROOT / "outputs"

FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

# Plot styling
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGDIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {FIGDIR / filename}")


def generate_retrieval_chart(df_ret):
    """Figure 1: Retrieval Performance Metrics (500 Questions)."""
    n_total = len(df_ret)
    hit1 = (df_ret["hit1"].astype(str).str.lower() == "true").mean() * 100
    hit3 = (df_ret["hit3"].astype(str).str.lower() == "true").mean() * 100
    hit5 = (df_ret["hit5"].astype(str).str.lower() == "true").mean() * 100
    succ = (df_ret["retrieval_success"].astype(str).str.lower() == "true").mean() * 100
    mrr = pd.to_numeric(df_ret["rr"], errors="coerce").fillna(0).mean()

    fig, ax = plt.subplots(figsize=(8.5, 5))
    metrics = ["Hit@1", "Hit@3", "Hit@5", "Retrieval\nSuccess"]
    vals = [hit1, hit3, hit5, succ]
    colors = ["#2563eb", "#3b82f6", "#60a5fa", "#10b981"]

    bars = ax.bar(metrics, vals, color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Metric Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"RAG Retrieval Performance — Full Benchmark (N={n_total} Questions)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y")

    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    ax.text(0.5, -0.18, f"Mean Reciprocal Rank (MRR): {mrr:.4f}  |  Evaluated on 250 Constitution + 250 Consumer Protection Questions",
            transform=ax.transAxes, ha="center", fontsize=9.5, style="italic", color="#334155")
    
    save_fig(fig, "01_retrieval_metrics_500.png")

    # Save table
    tab_df = pd.DataFrame([{
        "benchmark": f"500-Question Legal RAG (N={n_total})",
        "hit_at_1_pct": round(hit1, 2),
        "hit_at_3_pct": round(hit3, 2),
        "hit_at_5_pct": round(hit5, 2),
        "retrieval_success_pct": round(succ, 2),
        "mrr": round(mrr, 4)
    }])
    tab_df.to_csv(TABDIR / "table_retrieval_metrics_500.csv", index=False)


def generate_groundedness_chart(df_ans):
    """Figure 2: Groundedness Distribution (500 Questions)."""
    grounded = pd.to_numeric(df_ans["groundedness"], errors="coerce").dropna()
    avg_g = grounded.mean()
    med_g = grounded.median()
    above_70 = (grounded >= 70.0).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    counts, bins, patches = ax.hist(grounded, bins=20, range=(0, 100), color="#6366f1", edgecolor="#312e81", alpha=0.85, rwidth=0.88)
    
    ax.axvline(70.0, color="#dc2626", linestyle="--", linewidth=1.8, label=f"70% Threshold ({above_70:.1f}% of answers)")
    ax.axvline(avg_g, color="#d97706", linestyle="-", linewidth=1.8, label=f"Mean: {avg_g:.1f}%")
    ax.axvline(med_g, color="#059669", linestyle=":", linewidth=2.0, label=f"Median: {med_g:.1f}%")

    ax.set_xlabel("NLI Entailment Groundedness (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Generated Answers", fontsize=11, fontweight="bold")
    ax.set_title(f"Answer Groundedness Distribution (N={len(grounded)} Questions)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.grid(axis="y")
    ax.legend(frameon=True, facecolor="#f8fafc", loc="upper left", fontsize=10)

    ax.text(0.5, -0.20, "Groundedness measured via DeBERTa-v3 NLI entailment against retrieved legal contexts.\nAutomated NLI score is not a verified legal correctness guarantee.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#475569")

    save_fig(fig, "02_groundedness_distribution_500.png")

    # Save table
    tab_df = pd.DataFrame([{
        "sample_size": len(grounded),
        "mean_groundedness_pct": round(avg_g, 2),
        "median_groundedness_pct": round(med_g, 2),
        "pct_at_or_above_70": round(above_70, 2),
        "min_groundedness": round(grounded.min(), 2) if len(grounded) else 0,
        "max_groundedness": round(grounded.max(), 2) if len(grounded) else 0
    }])
    tab_df.to_csv(TABDIR / "table_answer_metrics_500.csv", index=False)


def generate_latency_chart(df_ret, df_ans=None):
    """Figure 3: Latency Distribution Analysis."""
    ret_lat = pd.to_numeric(df_ret["latency_seconds"], errors="coerce").dropna()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    if df_ans is not None and not df_ans.empty and "latency_seconds" in df_ans.columns:
        ans_lat = pd.to_numeric(df_ans["latency_seconds"], errors="coerce").dropna()
        data = [ret_lat, ans_lat]
        labels = [f"Retrieval Only\n(N={len(ret_lat)})", f"Full Generation + NLI\n(N={len(ans_lat)})"]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45,
                        boxprops=dict(facecolor="#93c5fd", color="#1e40af"),
                        medianprops=dict(color="#b91c1c", linewidth=2.0),
                        whiskerprops=dict(color="#1e40af"),
                        capprops=dict(color="#1e40af"))
        bp['boxes'][1].set_facecolor('#c7d2fe')
        bp['boxes'][1].set_edgecolor('#4338ca')
        ax.set_ylabel("Latency (Seconds)", fontsize=11, fontweight="bold")
        ax.set_title("Response Latency Distribution Across Pipeline Stages", fontsize=13, fontweight="bold", pad=12)
        
        r_med = ret_lat.median()
        a_med = ans_lat.median()
        ax.text(1, r_med + 0.5, f"Med: {r_med:.2f}s", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(2, a_med + 0.5, f"Med: {a_med:.2f}s", ha="center", fontsize=9.5, fontweight="bold")
    else:
        ax.hist(ret_lat, bins=25, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85, rwidth=0.88)
        med = ret_lat.median()
        mean = ret_lat.mean()
        ax.axvline(med, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Median: {med:.2f}s")
        ax.axvline(mean, color="#d97706", linestyle="-", linewidth=1.8, label=f"Mean: {mean:.2f}s")
        ax.set_xlabel("Retrieval Latency (Seconds)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count of Questions", fontsize=11, fontweight="bold")
        ax.set_title(f"Retrieval Latency Distribution (N={len(ret_lat)} Questions)", fontsize=13, fontweight="bold", pad=12)
        ax.legend(frameon=True, facecolor="#f8fafc")

    ax.grid(axis="y")
    save_fig(fig, "03_latency_distribution_500.png")


def generate_domain_comparison_chart(df_ret, df_ans=None):
    """Figure 4: Constitution vs. Consumer Protection Act Comparison."""
    const_ret = df_ret[df_ret["expected_source"].str.contains("constitution", case=False, na=False)]
    cp_ret = df_ret[df_ret["expected_source"].str.contains("consumer", case=False, na=False)]

    c_h1 = (const_ret["hit1"].astype(str).str.lower() == "true").mean() * 100
    c_h3 = (const_ret["hit3"].astype(str).str.lower() == "true").mean() * 100
    c_h5 = (const_ret["hit5"].astype(str).str.lower() == "true").mean() * 100
    c_mrr = pd.to_numeric(const_ret["rr"], errors="coerce").fillna(0).mean() * 100

    cp_h1 = (cp_ret["hit1"].astype(str).str.lower() == "true").mean() * 100
    cp_h3 = (cp_ret["hit3"].astype(str).str.lower() == "true").mean() * 100
    cp_h5 = (cp_ret["hit5"].astype(str).str.lower() == "true").mean() * 100
    cp_mrr = pd.to_numeric(cp_ret["rr"], errors="coerce").fillna(0).mean() * 100

    metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR (x100)"]
    c_vals = [c_h1, c_h3, c_h5, c_mrr]
    cp_vals = [cp_h1, cp_h3, cp_h5, cp_mrr]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - width/2, c_vals, width, label=f"Constitution of India (N={len(const_ret)})", color="#1e40af", edgecolor="#0f172a")
    b2 = ax.bar(x + width/2, cp_vals, width, label=f"Consumer Protection Act 2019 (N={len(cp_ret)})", color="#059669", edgecolor="#064e3b")

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Domain Retrieval Comparison (Balanced 250 vs 250 Questions)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="lower right", fontsize=10)
    ax.grid(axis="y")

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")

    save_fig(fig, "04_constitution_vs_consumer_protection_500.png")

    # Save domain comparison table
    tab_df = pd.DataFrame([
        {"Domain": "Constitution of India", "N": len(const_ret), "Hit@1 (%)": round(c_h1, 2), "Hit@3 (%)": round(c_h3, 2), "Hit@5 (%)": round(c_h5, 2), "MRR": round(c_mrr/100, 4)},
        {"Domain": "Consumer Protection Act 2019", "N": len(cp_ret), "Hit@1 (%)": round(cp_h1, 2), "Hit@3 (%)": round(cp_h3, 2), "Hit@5 (%)": round(cp_h5, 2), "MRR": round(cp_mrr/100, 4)}
    ])
    tab_df.to_csv(TABDIR / "table_domain_comparison_500.csv", index=False)


def generate_baseline_vs_500_chart(df_ret_500, df_ans_500=None):
    """Figure 5: Baseline 30-Question vs 500-Question Benchmark Comparison."""
    # Baseline 30-question stats from actual reports:
    # Retrieval: Hit@1=100%, Hit@3=100%, Hit@4=100%, MRR=1.0000
    # Answers: Groundedness=50.68%, Groundedness>=70%=36.67%, Source Match=100%
    b30_ret = {"Hit@1": 100.0, "Hit@3": 100.0, "Hit@5": 100.0, "MRR_x100": 100.0}
    
    n500 = len(df_ret_500)
    h1_500 = (df_ret_500["hit1"].astype(str).str.lower() == "true").mean() * 100
    h3_500 = (df_ret_500["hit3"].astype(str).str.lower() == "true").mean() * 100
    h5_500 = (df_ret_500["hit5"].astype(str).str.lower() == "true").mean() * 100
    mrr_500 = pd.to_numeric(df_ret_500["rr"], errors="coerce").fillna(0).mean() * 100

    metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR (x100)"]
    vals_30 = [b30_ret["Hit@1"], b30_ret["Hit@3"], b30_ret["Hit@5"], b30_ret["MRR_x100"]]
    vals_500 = [h1_500, h3_500, h5_500, mrr_500]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - width/2, vals_30, width, label="Baseline (N=30 Questions, Small Set)", color="#94a3b8", edgecolor="#334155")
    b2 = ax.bar(x + width/2, vals_500, width, label=f"New Benchmark (N={n500} Questions, Full Set)", color="#2563eb", edgecolor="#1e3a8a")

    ax.set_ylim(0, 120)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Metric Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Benchmark Comparison: Initial 30-Question Baseline vs. New 500-Question Dataset", fontsize=12.5, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="lower right", fontsize=10)
    ax.grid(axis="y")

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.text(0.5, -0.19, "Note: Baseline (N=30) consisted of hand-selected high-level queries; 500-question benchmark tests wide statutory coverage.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#475569")

    save_fig(fig, "05_baseline_30q_vs_500q_comparison.png")

    # Save comparison table
    tab_df = pd.DataFrame([
        {"Evaluation Run": "Baseline Test Set", "Sample Size (N)": 30, "Hit@1 (%)": 100.0, "Hit@3 (%)": 100.0, "Hit@5 (%)": 100.0, "MRR": 1.0000, "Notes": "Original 30-question development set"},
        {"Evaluation Run": "Scaled Legal Benchmark", "Sample Size (N)": n500, "Hit@1 (%)": round(h1_500, 2), "Hit@3 (%)": round(h3_500, 2), "Hit@5 (%)": round(h5_500, 2), "MRR": round(mrr_500/100, 4), "Notes": "500 unique questions across Constitution and Consumer Protection Act"}
    ])
    tab_df.to_csv(TABDIR / "table_baseline_vs_500_comparison.csv", index=False)


def generate_error_breakdown_chart(df_ret):
    """Figure 6: Failure & Error Category Breakdown."""
    n_total = len(df_ret)
    hit1_count = (df_ret["hit1"].astype(str).str.lower() == "true").sum()
    hit3_only = ((df_ret["hit3"].astype(str).str.lower() == "true") & (df_ret["hit1"].astype(str).str.lower() != "true")).sum()
    hit5_only = ((df_ret["hit5"].astype(str).str.lower() == "true") & (df_ret["hit3"].astype(str).str.lower() != "true")).sum()
    miss_count = (df_ret["hit5"].astype(str).str.lower() != "true").sum()

    labels = ["Rank 1 Hit", "Rank 2-3 Hit", "Rank 4-5 Hit", "Retrieval Miss (Rank >5)"]
    sizes = [hit1_count, hit3_only, hit5_only, miss_count]
    colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, texts, autotexts = ax.pie(
        [s for s in sizes if s > 0],
        labels=[l for l, s in zip(labels, sizes) if s > 0],
        autopct="%1.1f%%",
        startangle=140,
        colors=[c for c, s in zip(colors, sizes) if s > 0],
        wedgeprops=dict(edgecolor="#ffffff", linewidth=1.5),
        textprops=dict(fontsize=10)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title(f"Question-Level Retrieval Outcome Distribution (N={n_total})", fontsize=13, fontweight="bold", pad=12)
    save_fig(fig, "06_failures_and_errors_500.png")

    # Question types breakdown
    if "question_type" in df_ret.columns:
        type_perf = []
        for qtype, group in df_ret.groupby("question_type"):
            type_perf.append({
                "question_type": qtype,
                "count": len(group),
                "hit1_pct": round((group["hit1"].astype(str).str.lower() == "true").mean() * 100, 2),
                "hit5_pct": round((group["hit5"].astype(str).str.lower() == "true").mean() * 100, 2),
                "mrr": round(pd.to_numeric(group["rr"], errors="coerce").fillna(0).mean(), 4)
            })
        type_df = pd.DataFrame(type_perf).sort_values("count", ascending=False)
        type_df.to_csv(TABDIR / "table_question_type_performance_500.csv", index=False)


def main():
    print("=" * 70)
    print("GENERATING 500-QUESTION BENCHMARK FIGURES & TABLES")
    print("=" * 70)

    ret_csv = OUTPUTS / "RAG_RETRIEVAL_EVALUATION_500.csv"
    ans_csv = OUTPUTS / "RAG_ANSWER_EVALUATION_500.csv"

    if not ret_csv.exists():
        print(f"Error: {ret_csv} does not exist yet. Please run evaluate_rag_retrieval.py first.")
        return

    df_ret = pd.read_csv(ret_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df_ret)} retrieval results from {ret_csv}")

    df_ans = None
    if ans_csv.exists():
        try:
            df_ans = pd.read_csv(ans_csv, encoding="utf-8-sig")
            print(f"Loaded {len(df_ans)} answer results from {ans_csv}")
        except Exception as e:
            print(f"Warning loading answers CSV: {e}")

    generate_retrieval_chart(df_ret)
    generate_domain_comparison_chart(df_ret, df_ans)
    generate_baseline_vs_500_chart(df_ret, df_ans)
    generate_error_breakdown_chart(df_ret)
    generate_latency_chart(df_ret, df_ans)

    if df_ans is not None and not df_ans.empty and "groundedness" in df_ans.columns:
        generate_groundedness_chart(df_ans)

    print("\nAll figures generated in:", FIGDIR)
    print("All tables generated in:", TABDIR)


if __name__ == "__main__":
    main()

