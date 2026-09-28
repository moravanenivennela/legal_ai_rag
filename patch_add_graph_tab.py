with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add import
old_imports = "from fpdf import FPDF"
new_imports = "from fpdf import FPDF\nfrom streamlit_agraph import agraph, Node, Edge, Config"

if old_imports in content:
    content = content.replace(old_imports, new_imports, 1)
    print("Import added.")
else:
    print("WARNING: import anchor not found.")

# 2. Add a 4th tab
old_tabs = 'tab_chat, tab_dashboard, tab_about = st.tabs(["💬 Chat", "📊 Analytics Dashboard", "ℹ️ About This Project"])'
new_tabs = 'tab_chat, tab_dashboard, tab_about, tab_graph = st.tabs(["💬 Chat", "📊 Analytics Dashboard", "ℹ️ About This Project", "🕸️ Knowledge Graph"])'

if old_tabs in content:
    content = content.replace(old_tabs, new_tabs, 1)
    print("Tabs line updated.")
else:
    print("WARNING: tabs line not found.")

# 3. Append the graph tab content at the very end of the file
graph_tab_code = '''

with tab_graph:
    st.subheader("🕸️ Legal Knowledge Graph")
    st.caption("Illustrative diagram showing how the indexed legal domains relate. Not dynamically generated from live queries.")

    nodes = [
        Node(id="Indian Legal System", label="Indian Legal System", color="#D4AF37", size=25),
        Node(id="Constitution", label="Constitution of India", color="#2563EB", size=20),
        Node(id="Consumer Act", label="Consumer Protection Act 2019", color="#10B981", size=20),
        Node(id="Art 14", label="Article 14 (Equality)", color="#9333EA", size=15),
        Node(id="Art 21", label="Article 21 (Life & Liberty)", color="#9333EA", size=15),
        Node(id="Art 32", label="Article 32 (Writs)", color="#9333EA", size=15),
        Node(id="Art 246", label="Article 246 (Legislative Powers)", color="#9333EA", size=15),
        Node(id="Consumer Rights", label="Six Consumer Rights", color="#F59E0B", size=15),
        Node(id="Deficiency", label="Deficiency of Service", color="#F59E0B", size=15),
        Node(id="District Comm", label="District Commission", color="#F59E0B", size=15),
    ]
    edges = [
        Edge(source="Indian Legal System", target="Constitution"),
        Edge(source="Indian Legal System", target="Consumer Act"),
        Edge(source="Constitution", target="Art 14"),
        Edge(source="Constitution", target="Art 21"),
        Edge(source="Constitution", target="Art 32"),
        Edge(source="Constitution", target="Art 246"),
        Edge(source="Consumer Act", target="Consumer Rights"),
        Edge(source="Consumer Act", target="Deficiency"),
        Edge(source="Consumer Act", target="District Comm"),
    ]

    config = Config(width=900, height=500, directed=True, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)
'''

if content.rstrip().endswith('""")'):
    content = content.rstrip() + graph_tab_code
    print("Graph tab content appended.")
else:
    print("WARNING: file did not end as expected; appending anyway.")
    content = content.rstrip() + graph_tab_code

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
