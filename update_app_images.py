with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add shown_images session state init, right after uploaded_chunks init
old_init = '''if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = []'''
new_init = '''if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = []
if "shown_images" not in st.session_state:
    st.session_state.shown_images = set()'''

if old_init in content:
    content = content.replace(old_init, new_init, 1)
    print("Added shown_images init.")
else:
    print("WARNING: init block not found.")

# 2. Update the get_topic_image call to pass shown_images and track it
old_call = '''        topic_image_path = engine.get_topic_image(predicted_domain)
        if topic_image_path:
            st.image(topic_image_path, caption=f"Related topic: {predicted_domain.replace('_', ' ').title()}", width=350)'''
new_call = '''        topic_image_path = engine.get_topic_image(predicted_domain, st.session_state.shown_images)
        if topic_image_path:
            st.session_state.shown_images.add(topic_image_path)
            st.image(topic_image_path, caption=f"Related topic: {predicted_domain.replace('_', ' ').title()}", width=350)'''

if old_call in content:
    content = content.replace(old_call, new_call, 1)
    print("Updated get_topic_image call.")
else:
    print("WARNING: call block not found.")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
