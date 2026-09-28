with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

old_header = '''st.title("⚖️ Open Legal AI Assistant")
st.caption("Based on 'Legal AI for All: Reducing Perplexity & Boosting Accuracy in Normative Texts With Fine-Tuned LLMs and RAG'")'''

new_header = '''st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="background: linear-gradient(90deg, #D4AF37, #F4E4BC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 42px; margin-bottom: 0;">
        ⚖️ Nyaya AI — Legal Assistant
    </h1>
    <p style="color: #9CA3AF; font-size: 14px; margin-top: 4px;">
        Powered by Hybrid RAG + 4 Trained Deep Learning Models · Constitution of India · Consumer Protection Act, 2019
    </p>
</div>
""", unsafe_allow_html=True)'''

if old_header in content:
    content = content.replace(old_header, new_header, 1)
    print("Header updated.")
else:
    print("WARNING: header anchor not found, no changes made.")

old_init_marker = '''if "pending_query" not in st.session_state:
    st.session_state.pending_query = None'''

sample_chips = '''if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# --- Sample question chips (shown only on empty chat) ---
if not st.session_state.get("messages"):
    st.markdown("<p style='color:#9CA3AF; font-size:14px; margin-bottom:6px;'>Try asking:</p>", unsafe_allow_html=True)
    sample_qs = [
        "What are the six consumer rights under the Consumer Protection Act?",
        "What does Article 21 of the Constitution protect?",
        "Explain the writ jurisdiction under Article 32."
    ]
    chip_cols = st.columns(len(sample_qs))
    for i, q in enumerate(sample_qs):
        if chip_cols[i].button(q, key=f"sample_chip_{i}"):
            st.session_state.pending_query = q
            st.rerun()'''

if old_init_marker in content:
    content = content.replace(old_init_marker, sample_chips, 1)
    print("Sample chips added.")
else:
    print("WARNING: init marker not found, no changes made.")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
