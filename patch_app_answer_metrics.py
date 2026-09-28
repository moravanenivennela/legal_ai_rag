with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Compute metrics right after groundedness is checked, and render for NEW answers
old_generate_block = '''            with st.spinner("Checking groundedness..."):
                groundedness = engine.check_groundedness(full_response, contexts)'''

new_generate_block = '''            with st.spinner("Checking groundedness..."):
                groundedness = engine.check_groundedness(full_response, contexts)

            answer_metrics = engine.compute_answer_metrics(full_response, contexts)
            m_cols = st.columns(4)
            m_cols[0].metric("Accuracy", f"{answer_metrics['accuracy']:.1f}%")
            m_cols[1].metric("Precision", f"{answer_metrics['precision']:.1f}%")
            m_cols[2].metric("Recall", f"{answer_metrics['recall']:.1f}%")
            m_cols[3].metric("F1-score", f"{answer_metrics['f1']:.1f}%")'''

if old_generate_block in content:
    content = content.replace(old_generate_block, new_generate_block, 1)
    print("Live metrics added for new answers.")
else:
    print("WARNING: generate block anchor not found.")

# 2. Save answer_metrics into the message so it persists on rerun/history display
old_append = '''            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "min_distance": min_distance,
                "groundedness": groundedness,
                "followups": followups,
                "topic_image": topic_image_path,
                "predicted_domain": predicted_domain,
                "response_time_sec": elapsed,
                "baseline_answer": baseline_answer
            })'''

new_append = '''            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "min_distance": min_distance,
                "groundedness": groundedness,
                "followups": followups,
                "topic_image": topic_image_path,
                "predicted_domain": predicted_domain,
                "response_time_sec": elapsed,
                "baseline_answer": baseline_answer,
                "answer_metrics": answer_metrics
            })'''

if old_append in content:
    content = content.replace(old_append, new_append, 1)
    print("Metrics saved into message history.")
else:
    print("WARNING: append block anchor not found.")

# 3. Render saved metrics when redisplaying historical messages
old_history_render = '''            if msg["role"] == "assistant" and "groundedness" in msg and "response_time_sec" in msg:
                render_risk_scorecard(msg["groundedness"], msg.get("response_time_sec", 0.0))'''

new_history_render = '''            if msg["role"] == "assistant" and "groundedness" in msg and "response_time_sec" in msg:
                render_risk_scorecard(msg["groundedness"], msg.get("response_time_sec", 0.0))

            if msg.get("answer_metrics"):
                am = msg["answer_metrics"]
                hm_cols = st.columns(4)
                hm_cols[0].metric("Accuracy", f"{am['accuracy']:.1f}%")
                hm_cols[1].metric("Precision", f"{am['precision']:.1f}%")
                hm_cols[2].metric("Recall", f"{am['recall']:.1f}%")
                hm_cols[3].metric("F1-score", f"{am['f1']:.1f}%")'''

if old_history_render in content:
    content = content.replace(old_history_render, new_history_render, 1)
    print("Historical metrics rendering added.")
else:
    print("WARNING: history render anchor not found.")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
