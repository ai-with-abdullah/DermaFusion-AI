import matplotlib.pyplot as plt
import numpy as np

# --- GRAPH 1: Training Progression (EVA Small baseline) ---
epochs = [2, 5, 10, 17, 23, 25]
auc = [0.9569, 0.9749, 0.9825, 0.9839, 0.9831, 0.9831]
bal_acc = [0.6283, 0.7241, 0.7596, 0.7771, 0.7583, 0.7612]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Macro AUC', color='steelblue')
ax1.plot(epochs, auc, 'o-', color='steelblue', label='Macro AUC')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.axvline(x=17, color='red', linestyle='--', alpha=0.5, label='Best epoch (17)')

ax2 = ax1.twinx()
ax2.set_ylabel('Balanced Accuracy', color='darkorange')
ax2.plot(epochs, bal_acc, 's--', color='darkorange', label='Balanced Accuracy')
ax2.tick_params(axis='y', labelcolor='darkorange')

plt.title('EVA-02 Small Training Progression')
fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.15))
plt.tight_layout()
plt.savefig('graph1_training_progression.png', dpi=300)
plt.show()

# --- GRAPH 2: Per-Class AUC of Final Model ---
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
per_class_auc = [0.997, 0.999, 0.989, 0.999, 0.972, 0.980, 1.000]
colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12','#1abc9c','#e67e22']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(classes, per_class_auc, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylim(0.95, 1.005)
ax.set_ylabel('AUC (one-vs-rest)')
ax.set_title('Per-Class AUC — DermaFusion-AI Final Model (Run 2)')
ax.axhline(y=0.9908, color='red', linestyle='--', label='Macro Avg AUC = 0.9908')
for bar, val in zip(bars, per_class_auc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig('graph2_per_class_auc.png', dpi=300)
plt.show()
