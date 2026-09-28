with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

old_marker = '''#### Benchmark result
Retrieval accuracy improved from **85.7% to 100%** after adding the query domain classifier as a second guardrail layer, measured on a custom difficulty-tiered benchmark.
""")'''

new_content_block = '''#### Benchmark result
Retrieval accuracy improved from **85.7% to 100%** after adding the query domain classifier as a second guardrail layer, measured on a custom difficulty-tiered benchmark.
""")

    st.markdown("#### 📐 Formal Evaluation Metrics")
    st.caption("Measured via cross-validation (classifier), held-out validation split (CNN), and a 16-question labeled test set (guardrail).")

    eval_data = {
        "Model": ["Query Domain Classifier", "Document Image Classifier (CNN)", "Retrieval Guardrail"],
        "Accuracy": ["86.7%", "85.9%", "100.0%*"],
        "Precision (macro)": ["86.6%", "87.9%", "100.0%*"],
        "Recall (macro)": ["86.7%", "88.7%", "100.0%*"],
        "F1-score (macro)": ["86.6%", "85.8%", "100.0%*"],
    }
    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)
    st.caption("*Guardrail evaluated on a small (16-example) labeled test set — a larger adversarial test set would be needed for a stronger production claim.")
'''

if old_marker in content:
    content = content.replace(old_marker, new_content_block, 1)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Metrics table added to About tab.")
else:
    print("WARNING: anchor not found — no changes made.")
