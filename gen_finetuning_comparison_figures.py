import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

with open("llm_results_incremental.json") as f:
    rag_results = json.load(f)
with open("rag_vs_norag_incremental.json") as f:
    norag_results = json.load(f)

BASE_MODEL = "llama3.2:3b"
FT_MODEL = "legal-ai-finetuned"

if FT_MODEL not in rag_results or FT_MODEL not in norag_results:
    print(f"ERROR: '{FT_MODEL}' not found in one of the result files.")
    print("rag_results keys:", list(rag_results.keys()))
    print("norag_results keys:", list(norag_results.keys()))
    exit(1)

# ============================================================
# 1. The 2x2 grid: Base vs Fine-tuned, No-RAG vs RAG
# ============================================================
base_norag_f1 = norag_results.get(BASE_MODEL, {}).get("norag", {}).get("f1")
base_rag_f1 = rag_results.get(BASE_MODEL, {}).get("f1")
ft_norag_f1 = norag_results[FT_MODEL]["norag"]["f1"]
ft_rag_f1 = rag_results[FT_MODEL]["f1"]

print(f"Base model ({BASE_MODEL}):")
print(f"  No-RAG F1: {base_norag_f1}")
print(f"  RAG F1:    {base_rag_f1}")
print(f"Fine-tuned model ({FT_MODEL}):")
print(f"  No-RAG F1: {ft_norag_f1}")
print(f"  RAG F1:    {ft_rag_f1}")

if base_norag_f1 is not None and base_rag_f1 is not None:
    fig, ax = plt.subplots(figsize=(9, 7))
    data_2x2 = np.array([[base_norag_f1, base_rag_f1], [ft_norag_f1, ft_rag_f1]])
    im = ax.imshow(data_2x2, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks([0,1]); ax.set_xticklabels(["No RAG", "With RAG"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Base Model", "Fine-tuned Model"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{data_2x2[i,j]:.1f}%", ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white" if data_2x2[i,j] > 55 else "black")
    ax.set_title(f"Fine-tuning \u00d7 RAG: F1-score Comparison\n({BASE_MODEL} base vs. fine-tuned variant)", fontweight="bold")
    plt.colorbar(im, label="F1-score (%)")
    plt.tight_layout()
    plt.savefig("outputs/combined_analysis/finetuning_rag_2x2_matrix.png")
    plt.close()
    print("Saved: finetuning_rag_2x2_matrix.png")

    # Grouped bar version of the same data (easier to read exact values)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [base_norag_f1, base_rag_f1], width, label="Base Model", color="#94a3b8")
    bars2 = ax.bar(x + width/2, [ft_norag_f1, ft_rag_f1], width, label="Fine-tuned Model", color="#c5a059")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,3), textcoords="offset points", ha='center', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(["No RAG", "With RAG"])
    ax.set_ylabel("F1-score (%)"); ax.set_ylim(0,100)
    ax.set_title("Effect of Fine-tuning and RAG on Answer Quality", fontweight="bold")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/combined_analysis/finetuning_rag_grouped_bar.png")
    plt.close()
    print("Saved: finetuning_rag_grouped_bar.png")
else:
    print(f"WARNING: '{BASE_MODEL}' missing from one of the result files — only plotting fine-tuned model's own No-RAG vs RAG.")

# ============================================================
# 2. Fine-tuned model: All metrics, RAG vs No-RAG
# ============================================================
metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
ft_rag_vals = [rag_results[FT_MODEL][m] for m in metrics_to_plot]
ft_norag_vals = [norag_results[FT_MODEL]["norag"][m] for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(9,6))
bars1 = ax.bar(x - width/2, ft_rag_vals, width, label="Fine-tuned + RAG", color="#10b981")
bars2 = ax.bar(x + width/2, ft_norag_vals, width, label="Fine-tuned, No RAG", color="#ef4444")
for bars in [bars1, bars2]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,3), textcoords="offset points", ha='center', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metric_labels)
ax.set_ylabel("Score (%)"); ax.set_ylim(0,105)
ax.set_title("Fine-tuned Model: RAG vs. No-RAG Performance", fontweight="bold")
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/combined_analysis/finetuned_model_rag_vs_norag.png")
plt.close()
print("Saved: finetuned_model_rag_vs_norag.png")

# ============================================================
# 3. Full comparison: Base(RAG), Base(No-RAG), FT(RAG), FT(No-RAG)
# ============================================================
labels = ["Base\n(No RAG)", "Base\n(RAG)", "Fine-tuned\n(No RAG)", "Fine-tuned\n(RAG)"]
values = [base_norag_f1 or 0, base_rag_f1 or 0, ft_norag_f1, ft_rag_f1]
colors = ["#94a3b8", "#3b82f6", "#f59e0b", "#10b981"]

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(labels, values, color=colors)
for bar, val in zip(bars, values):
    ax.annotate(f'{val:.1f}%', xy=(bar.get_x()+bar.get_width()/2, val), xytext=(0,3),
                textcoords="offset points", ha='center', fontweight='bold')
ax.set_ylabel("F1-score (%)"); ax.set_ylim(0,100)
ax.set_title("Complete Comparison: Fine-tuning \u00d7 RAG Configurations\n(mirrors paper's core experimental design)", fontweight="bold")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/combined_analysis/full_finetuning_rag_comparison.png")
plt.close()
print("Saved: full_finetuning_rag_comparison.png")

print("\nAll fine-tuning comparison figures saved to outputs/combined_analysis/")
