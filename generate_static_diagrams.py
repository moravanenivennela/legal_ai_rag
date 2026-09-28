import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#0d1117'
plt.rcParams['savefig.facecolor'] = '#0d1117'

# --- Diagram 1: Legal Domain Relationship Map (static network) ---
G = nx.DiGraph()
edges = [
    ("Indian Legal System", "Constitution"),
    ("Indian Legal System", "Consumer Act"),
    ("Constitution", "Art 14"),
    ("Constitution", "Art 21"),
    ("Constitution", "Art 32"),
    ("Constitution", "Art 246"),
    ("Consumer Act", "Consumer Rights"),
    ("Consumer Act", "Deficiency"),
    ("Consumer Act", "District Comm"),
]
G.add_edges_from(edges)

node_colors_map = {
    "Indian Legal System": "#c5a059",
    "Constitution": "#3b82f6",
    "Consumer Act": "#10b981",
    "Art 14": "#9333EA", "Art 21": "#9333EA", "Art 32": "#9333EA", "Art 246": "#9333EA",
    "Consumer Rights": "#f59e0b", "Deficiency": "#f59e0b", "District Comm": "#f59e0b",
}
node_colors = [node_colors_map.get(n, "#888888") for n in G.nodes()]
node_sizes = [2500 if n == "Indian Legal System" else 1800 if n in ["Constitution", "Consumer Act"] else 1200 for n in G.nodes()]

pos = nx.spring_layout(G, seed=42, k=1.2)

fig, ax = plt.subplots(figsize=(10, 7))
nx.draw_networkx_edges(G, pos, edge_color="#444d56", arrows=True, arrowsize=15, width=1.5, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="#c5a059", linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", font_weight="bold", ax=ax)

ax.set_title("Legal Domain Relationship Map", color="#F4E4BC", fontsize=16, fontweight="bold", pad=20)
ax.axis("off")
plt.tight_layout()
plt.savefig("static_knowledge_graph.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved static_knowledge_graph.png")

# --- Diagram 2: Pipeline Flow Diagram ---
fig, ax = plt.subplots(figsize=(10, 6))
stages = [
    ("User Query", "#9CA3AF"),
    ("Query Classifier\n(DL Guardrail)", "#9333EA"),
    ("Hybrid Retrieval\n(BM25 + Dense + RRF)", "#3b82f6"),
    ("Cross-Encoder\nReranker", "#3b82f6"),
    ("LLM Generation\n(Ollama)", "#c5a059"),
    ("NLI Groundedness\nScorer", "#f59e0b"),
    ("Final Answer", "#10b981"),
]

y = 0.5
x_positions = [i * 1.5 for i in range(len(stages))]

for i, (label, color) in enumerate(stages):
    box = mpatches.FancyBboxPatch((x_positions[i], y - 0.15), 1.2, 0.3,
                                    boxstyle="round,pad=0.02", facecolor=color, edgecolor="white", linewidth=1)
    ax.add_patch(box)
    ax.text(x_positions[i] + 0.6, y, label, ha="center", va="center", fontsize=8, color="black", fontweight="bold")
    if i < len(stages) - 1:
        ax.annotate("", xy=(x_positions[i+1], y), xytext=(x_positions[i] + 1.2, y),
                    arrowprops=dict(arrowstyle="->", color="#c5a059", lw=1.5))

ax.set_xlim(-0.3, x_positions[-1] + 1.5)
ax.set_ylim(0, 1)
ax.set_title("Nyaya AI — Processing Pipeline", color="#F4E4BC", fontsize=16, fontweight="bold", pad=20)
ax.axis("off")
plt.tight_layout()
plt.savefig("static_pipeline_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved static_pipeline_diagram.png")
