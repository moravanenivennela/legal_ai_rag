with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove the chat_input capture from inside tab_chat
old_inside_tab = '''    user_query = st.chat_input("Ask a legal question based on your indexed normative documents...")
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None

'''
if old_inside_tab in content:
    content = content.replace(old_inside_tab, "", 1)
    print("Removed chat_input from inside tab.")
else:
    print("WARNING: inside-tab block not found.")

# 2. Add it at top level, right before the tabs are created
old_tabs_line = 'tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Analytics Dashboard"])'
new_tabs_line = '''user_query = st.chat_input("Ask a legal question based on your indexed normative documents...")
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Analytics Dashboard"])'''

if old_tabs_line in content:
    content = content.replace(old_tabs_line, new_tabs_line, 1)
    print("Added chat_input at top level.")
else:
    print("WARNING: tabs line not found.")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
