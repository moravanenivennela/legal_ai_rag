import numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
os.makedirs('outputs/plots', exist_ok=True)
cm = np.load('outputs/confusion_matrix.npy')
labels = ['Constitution', 'Consumer Protection', 'Out-of-Domain']
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=True, annot_kws={'size': 14})
plt.title('Fig 3. Domain Guardrail 3-Class Confusion Matrix', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
plt.ylabel('Actual Class', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrix.png', dpi=300)
print('Confusion Matrix saved to outputs/plots/confusion_matrix.png')
