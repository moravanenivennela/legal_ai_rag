with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

old_marker = 'st.caption("Illustrative diagram showing how the indexed legal domains relate. Not dynamically generated from live queries.")'

new_marker = '''st.caption("Illustrative diagram showing how the indexed legal domains relate. Not dynamically generated from live queries.")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image("static_knowledge_graph.png", caption="Static Legal Domain Map", use_container_width=True)
    with img_col2:
        st.image("static_pipeline_diagram.png", caption="Processing Pipeline Overview", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Interactive Version")'''

if old_marker in content:
    content = content.replace(old_marker, new_marker, 1)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Static images embedded.")
else:
    print("WARNING: marker not found.")
