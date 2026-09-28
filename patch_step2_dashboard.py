with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add time import and tracking list init
old_imports = '''import streamlit as st
from datetime import datetime
from rag_engine import LegalRAGEngine
import fitz'''

new_imports = '''import streamlit as st
from datetime import datetime
import time
from rag_engine import LegalRAGEngine
import fitz'''

if old_imports in content:
    content = content.replace(old_imports, new_imports, 1)
    print("Imports updated.")
else:
    print("WARNING: imports anchor not found.")

# 2. Wrap the whole app body in tabs, right after the branded header block
old_tab_anchor = '''st.sidebar.header("⚙️ System Configuration")'''
new_tab_anchor = '''tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Analytics Dashboard"])

if "query_log" not in st.session_state:
    st.session_state.query_log = []

st.sidebar.header("⚙️ System Configuration")'''

if old_tab_anchor in content:
    content = content.replace(old_tab_anchor, new_tab_anchor, 1)
    print("Tabs added.")
else:
    print("WARNING: tab anchor not found.")

# 3. Wrap the main chat rendering block inside "with tab_chat:"
old_chat_start = '''# --- Render historical chat messages from session state ---
for idx, msg in enumerate(st.session_state.messages):'''
new_chat_start = '''with tab_chat:
 # --- Render historical chat messages from session state ---
 for idx, msg in enumerate(st.session_state.messages):'''

if old_chat_start in content:
    content = content.replace(old_chat_start, new_chat_start, 1)
    print("Chat block wrapped (start).")
else:
    print("WARNING: chat start anchor not found.")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
