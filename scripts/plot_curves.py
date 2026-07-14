import json, matplotlib.pyplot as plt, numpy as np, os

base = '/home/rukiye/road_damage_project/models/'
out  = '/home/rukiye/road_damage_project/outputs/figures/'
os.makedirs(out, exist_ok=True)

with open(base + 'history_unet.json') as f: hunet = json.load(f)
with open(base + 'history_enet.json') as f: henet = json.load(f)
with open(base + 'history_768.json') as f: h768 = json.load(f)
with open(base + 'val_losses.json') as f: vl = json.load(f)

TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE  = 10

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(wspace=0.3, hspace=0.35)

labels = [['(a)', '(b)'], ['(c)', '(d)']]

# (a) Sol üst: Validation IoU
ax = axes[0][0]
ax.plot([e['val_iou'] for e in henet], label='ENet',         color='blue',   linewidth=2, linestyle='-.')
ax.plot([e['val_iou'] for e in hunet], label='U-Net',        color='orange', linewidth=2, linestyle='--')
ax.plot([e['val_iou'] for e in h768],  label='SegFormer-B2', color='green',  linewidth=2, linestyle='-')
ax.set_title('(a) Validation IoU', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
ax.set_ylabel('IoU', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.legend(fontsize=TICK_SIZE)
ax.grid(True, alpha=0.3)

# (b) Sağ üst: Train vs Val IoU
ax = axes[0][1]
ax.plot([e['train_iou'] for e in henet], label='Train (ENet)',         color='blue',   linewidth=1.5, linestyle='-.')
ax.plot([e['val_iou']   for e in henet], label='Val (ENet)',           color='blue',   linewidth=1.5, linestyle=':')
ax.plot([e['train_iou'] for e in hunet], label='Train (U-Net)',        color='orange', linewidth=1.5, linestyle='--')
ax.plot([e['val_iou']   for e in hunet], label='Val (U-Net)',          color='orange', linewidth=1.5, linestyle=':')
ax.plot([e['train_iou'] for e in h768],  label='Train (SegFormer-B2)', color='green',  linewidth=2,   linestyle='-')
ax.plot([e['val_iou']   for e in h768],  label='Val (SegFormer-B2)',   color='green',  linewidth=2,   linestyle='--')
ax.set_title('(b) Train vs Validation IoU', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
ax.set_ylabel('IoU', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) Sol alt: Final F1 bar
ax = axes[1][0]
models = ['ENet', 'U-Net', 'SegFormer-B2']
f1s = [0.4773, 0.6582, 0.8127]
colors = ['blue', 'orange', 'green']
bars = ax.bar(models, f1s, color=colors, alpha=0.8, width=0.5)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.4f}', ha='center', fontsize=TICK_SIZE, fontweight='bold')
ax.set_title('(c) Final Validation F1', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_ylabel('F1 Skoru', fontsize=LABEL_SIZE)
ax.set_ylim(0, 1.0)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3, axis='y')

# (d) Sağ alt: Final Loss bar
ax = axes[1][1]
losses = [vl['enet'], vl['unet'], vl['segformer']]
bars = ax.bar(models, losses, color=colors, alpha=0.8, width=0.5)
for bar, val in zip(bars, losses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=TICK_SIZE, fontweight='bold')
ax.set_title('(d) Final Validation Loss', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_ylabel('Loss', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3, axis='y')

plt.savefig(out + 'training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Kaydedildi!')
