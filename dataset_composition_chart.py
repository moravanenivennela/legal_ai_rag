"""
Dataset composition chart: shows how many questions/documents fall into each
legal domain. Edit COMPOSITION with your real counts.
"""
import matplotlib.pyplot as plt

COMPOSITION = {
    "Consumer Protection": 15,
    "Constitution": 15,
    "Out of Domain": 45,
}

labels = list(COMPOSITION.keys())
sizes = list(COMPOSITION.values())
colors = ["#3b82f6", "#10b981", "#f59e0b"]

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct=lambda p: f"{p:.1f}%\n(n={int(round(p * sum(sizes) / 100))})",
    colors=colors, startangle=90, textprops={"fontsize": 10, "fontweight": "bold"}
)
ax.set_title("Test Set Composition by Legal Domain", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/dataset_composition_chart.png", dpi=150)
plt.close()

print("Saved: outputs/dataset_composition_chart.png")
