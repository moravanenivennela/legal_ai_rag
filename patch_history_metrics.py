with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

old_block = '''            if "groundedness" in msg and msg["groundedness"] is not None:
                g_score = msg["groundedness"]
                if g_score >= 70:
                    st.markdown(f"✅ **Groundedness: {g_score:.1f}%** — well-supported by retrieved context")
                elif g_score >= 45:
                    st.markdown(f"⚠️ **Groundedness: {g_score:.1f}%** — partially supported, verify key claims")
                else:
                    st.markdown(f"🚫 **Groundedness: {g_score:.1f}%** — low support, possible hallucination")
            if msg.get("topic_image"):'''

new_block = '''            if "groundedness" in msg and msg["groundedness"] is not None:
                g_score = msg["groundedness"]
                if g_score >= 70:
                    st.markdown(f"✅ **Groundedness: {g_score:.1f}%** — well-supported by retrieved context")
                elif g_score >= 45:
                    st.markdown(f"⚠️ **Groundedness: {g_score:.1f}%** — partially supported, verify key claims")
                else:
                    st.markdown(f"🚫 **Groundedness: {g_score:.1f}%** — low support, possible hallucination")

            if msg.get("answer_metrics"):
                am = msg["answer_metrics"]
                hm_cols = st.columns(4)
                hm_cols[0].metric("Accuracy", f"{am['accuracy']:.1f}%")
                hm_cols[1].metric("Precision", f"{am['precision']:.1f}%")
                hm_cols[2].metric("Recall", f"{am['recall']:.1f}%")
                hm_cols[3].metric("F1-score", f"{am['f1']:.1f}%")

            if msg.get("topic_image"):'''

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Historical metrics rendering added.")
else:
    print("STILL NOT FOUND — will need manual inspection.")
