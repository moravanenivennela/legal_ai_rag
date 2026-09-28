with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

old_append = '''            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "min_distance": min_distance,
                "groundedness": groundedness,
                "followups": followups,
                "topic_image": topic_image_path,
                "predicted_domain": predicted_domain
            })'''

new_append = '''            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "min_distance": min_distance,
                "groundedness": groundedness,
                "followups": followups,
                "topic_image": topic_image_path,
                "predicted_domain": predicted_domain,
                "answer_metrics": answer_metrics
            })'''

if old_append in content:
    content = content.replace(old_append, new_append, 1)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Append block fixed.")
else:
    print("STILL NOT FOUND.")
