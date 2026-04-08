"""Figure 4B/C: Tissue-level analysis
B: Per-tissue Pearson boxplot across species
C: Mean expression vs prediction Pearson (tissue difficulty)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

PDIR = Path('/home/hjd/RNAi/results/predictions_20kb')
FDIR = Path('/home/hjd/RNAi/results/paper/figures')
FDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 11, 'ytick.labelsize': 12,
    'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
})

TISSUES = [
    'Adult_Brain', 'Adult_Head', 'Adult_Midgut', 'Adult_Hindgut',
    'Adult_FatBody', 'Adult_MalpighianTubule', 'Adult_Carcass',
    'Adult_Ovary', 'Adult_Testis',
    'Larval_Hindgut', 'Larval_FatBody', 'Larval_MalpighianTubule',
    'Larval_Midgut', 'Larval_Carcass',
]

SP_DISPLAY = {
    'tribolium': 'T.cas', 'drosophila': 'D.mel', 'silkworm': 'B.mor',
    'pxyl': 'P.xyl', 'apis': 'A.pis', 'nlug': 'N.lug', 'lmig': 'L.mig',
    'harm': 'H.arm', 'csup': 'C.sup', 'focc': 'F.occ', 'agla': 'A.gla',
    'ldec': 'L.dec',
}

TISSUE_SHORT = {t: t.replace('Adult_', 'A.').replace('Larval_', 'L.').replace('MalpighianTubule', 'MalTub')
                for t in TISSUES}

C_OURS = '#D62728'

# load per-species per-tissue Pearson
print('Computing per-species per-tissue Pearson...')
tissue_sp = {}  # {tissue: {species: pearson}}
tissue_stats = {}  # {tissue: (mean_expr, n_genes, pearson_all)}

all_csvs = sorted(PDIR.glob('*_predictions.csv'))
for f in all_csvs:
    sp = f.stem.replace('_predictions', '')
    df = pd.read_csv(f)
    for t in TISSUES:
        tc, tp = f'{t}_true', f'{t}_pred'
        if tc not in df.columns or tp not in df.columns:
            continue
        v = df[[tc, tp]].dropna()
        if len(v) < 20:
            continue
        r = pearsonr(v[tc], v[tp])[0]
        tissue_sp.setdefault(t, {})[sp] = r
        # accumulate for tissue stats
        if t not in tissue_stats:
            tissue_stats[t] = {'trues': [], 'preds': []}
        tissue_stats[t]['trues'].append(v[tc].values)
        tissue_stats[t]['preds'].append(v[tp].values)

# compute tissue-level aggregated stats
tissue_agg = {}
for t in TISSUES:
    if t not in tissue_stats:
        continue
    yt = np.concatenate(tissue_stats[t]['trues'])
    yp = np.concatenate(tissue_stats[t]['preds'])
    tissue_agg[t] = {
        'pearson': pearsonr(yt, yp)[0],
        'mean_expr': np.mean(yt),
        'n_genes': len(yt),
        'nonzero_frac': np.mean(yt > 0.1),
    }

# sort tissues by median Pearson across species
tissue_order = sorted(tissue_sp.keys(),
    key=lambda t: np.median(list(tissue_sp[t].values())))

print(f'  {len(tissue_order)} tissues, {len(all_csvs)} species')

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.35})

# Panel B: Boxplot of per-tissue Pearson across species
bp_data = [list(tissue_sp[t].values()) for t in tissue_order]
bp = ax1.boxplot(bp_data, vert=False, patch_artist=True,
                 widths=0.6, showfliers=True,
                 flierprops={'ms': 4, 'alpha': 0.5},
                 medianprops={'color': C_OURS, 'lw': 2})
for patch in bp['boxes']:
    patch.set_facecolor('#E8E8E8')
    patch.set_edgecolor('#888')

# overlay individual species points
for i, t in enumerate(tissue_order):
    vals = list(tissue_sp[t].values())
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
    ax1.scatter(vals, [i + 1 + j for j in jitter], s=18, alpha=0.6,
                color='#1F77B4', zorder=5, edgecolors='none')
    # median label
    med = np.median(vals)
    ax1.text(med, i + 1.35, f'{med:.3f}', ha='center', fontsize=8, color='#555')

ax1.set_yticks(range(1, len(tissue_order) + 1))
ax1.set_yticklabels([TISSUE_SHORT[t] for t in tissue_order])
ax1.set_xlabel('Pearson Correlation')
ax1.set_title('B  Per-tissue Pearson across 12 Species', loc='left', fontweight='bold')
ax1.grid(axis='x', alpha=0.15)

# Panel C: Mean expression vs tissue Pearson (difficulty scatter)
ts_with_agg = [t for t in tissue_order if t in tissue_agg]
x_vals = [tissue_agg[t]['nonzero_frac'] for t in ts_with_agg]
y_vals = [tissue_agg[t]['pearson'] for t in ts_with_agg]
sizes = [tissue_agg[t]['n_genes'] / 500 for t in ts_with_agg]

ax2.scatter(x_vals, y_vals, s=sizes, c=y_vals, cmap='RdYlGn', vmin=0.5, vmax=0.8,
            edgecolors='#555', linewidths=0.8, alpha=0.85, zorder=5)

for t, x, y in zip(ts_with_agg, x_vals, y_vals):
    ax2.annotate(TISSUE_SHORT[t], (x, y), fontsize=8.5, ha='center',
                 xytext=(0, 10), textcoords='offset points', color='#333')

ax2.set_xlabel('Non-zero Expression Fraction')
ax2.set_ylabel('Pearson Correlation (all species pooled)')
ax2.set_title('C  Tissue Difficulty vs Expression Coverage', loc='left', fontweight='bold')
ax2.grid(alpha=0.15)

# size legend
for s_val, s_label in [(10000, '10k'), (20000, '20k'), (30000, '30k')]:
    ax2.scatter([], [], s=s_val/500, c='#ccc', edgecolors='#888', label=f'{s_label} genes')
ax2.legend(loc='lower right', fontsize=9, framealpha=0.8, title='Sample size', title_fontsize=9)

plt.tight_layout()
print('Saving...')
for ext in ['png', 'pdf']:
    fig.savefig(FDIR / f'fig4bc_tissue_analysis.{ext}', dpi=300, bbox_inches='tight')
print(f'Done: {FDIR}/fig4bc_tissue_analysis.png')
plt.close()

