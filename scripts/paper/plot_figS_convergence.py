"""Supplementary Figure: Training convergence curves
4 models: InsectExpress 20kb, InsectExpress 50kb, Vanilla Enformer, Xpresso
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOGDIR = Path('/home/hjd/RNAi/logs')
FDIR = Path('/home/hjd/RNAi/results/paper/figures')
FDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14,
    'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_20KB = '#D62728'
C_50KB = '#FF7F0E'
C_VE = '#1F77B4'
C_XP = '#AAAAAA'


def parse_v2_log(path):
    # format: Epoch X/150 (LR: ...) then next line: Overall Pearson: 0.XXXX
    epochs, pearsons = [], []
    lines = path.read_text().splitlines()
    cur_epoch = None
    for line in lines:
        m = re.search(r'Epoch (\d+)/150', line)
        if m:
            cur_epoch = int(m.group(1))
        m2 = re.search(r'Overall Pearson: ([0-9.]+)', line)
        if m2 and cur_epoch is not None:
            epochs.append(cur_epoch)
            pearsons.append(float(m2.group(1)))
            cur_epoch = None
    return np.array(epochs), np.array(pearsons)


def parse_dl_log(path, model_name):
    # format: [Model] Epoch X/150 | ... | Val Pearson=0.XXXX
    epochs, pearsons = [], []
    for line in path.read_text().splitlines():
        if f'[{model_name}]' not in line:
            continue
        m = re.search(r'Epoch\s+(\d+)/150.*Val Pearson=([0-9.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            pearsons.append(float(m.group(2)))
    return np.array(epochs), np.array(pearsons)


# parse all logs
e20, p20 = parse_v2_log(LOGDIR / 'v2_20kb_20260319_092727.log')
e50, p50 = parse_v2_log(LOGDIR / 'v2_50kb_64x_12sp_splcluster_20260314_110114.log')
eve, pve = parse_dl_log(LOGDIR / 'dl_baselines_64x_12sp_20260314_110116.log', 'Vanilla_Enformer')
exp, pxp = parse_dl_log(LOGDIR / 'dl_baselines_64x_12sp_20260314_110116.log', 'Xpresso')

print(f'20kb: {len(e20)} points, best={p20.max():.4f} @e{e20[np.argmax(p20)]}')
print(f'50kb: {len(e50)} points, best={p50.max():.4f} @e{e50[np.argmax(p50)]}')
print(f'V.Enf: {len(eve)} points, best={pve.max():.4f} @e{eve[np.argmax(pve)]}')
print(f'Xpresso: {len(exp)} points, best={pxp.max():.4f} @e{exp[np.argmax(pxp)]}')

# ---- Figure ----
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(e20, p20, '-o', color=C_20KB, ms=5, lw=2.2, label='InsectExpress (20kb)', zorder=10)
ax.plot(e50, p50, '-s', color=C_50KB, ms=4, lw=1.8, label='InsectExpress (50kb)', zorder=8)
ax.plot(eve, pve, '-^', color=C_VE, ms=3, lw=1.5, label='Vanilla Enformer', zorder=6, alpha=0.8)
ax.plot(exp, pxp, '-d', color=C_XP, ms=3, lw=1.3, label='Xpresso', zorder=5, alpha=0.7)

# mark best points
for e, p, c, lbl in [(e20, p20, C_20KB, '20kb'), (e50, p50, C_50KB, '50kb'),
                       (eve, pve, C_VE, 'V.Enf'), (exp, pxp, C_XP, 'Xpr')]:
    bi = np.argmax(p)
    ax.scatter(e[bi], p[bi], s=120, c=c, edgecolors='white', linewidths=2, zorder=15, marker='*')
    ax.annotate(f'{p[bi]:.3f}', (e[bi], p[bi]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold', color=c)

# warmup region
ax.axvspan(0, 15, alpha=0.05, color='orange', zorder=0)
ax.text(7.5, ax.get_ylim()[0] + 0.02, 'warmup', ha='center', fontsize=9, color='#CC8800', alpha=0.7)

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Pearson Correlation')
ax.set_title('Training Convergence Comparison', fontweight='bold')
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.grid(alpha=0.15)
ax.set_xlim(-2, 150)
ax.set_ylim(0.05, 0.75)

plt.tight_layout()
print('Saving...')
for ext in ['png', 'pdf']:
    fig.savefig(FDIR / f'figS_convergence.{ext}', dpi=300, bbox_inches='tight')
print(f'Done: {FDIR}/figS_convergence.png')
plt.close()

