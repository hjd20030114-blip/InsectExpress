import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.family': 'sans-serif',
})
PRED_DIR = Path('/home/hjd/RNAi/results/predictions_20kb')
OUT_DIR = Path('/home/hjd/RNAi/results/paper/figures')

TISSUES = [
    'Adult_Brain', 'Adult_Head', 'Adult_Midgut', 'Adult_Hindgut',
    'Adult_FatBody', 'Adult_MalpighianTubule', 'Adult_Carcass',
    'Adult_Ovary', 'Adult_Testis', 'Larval_Hindgut', 'Larval_FatBody',
    'Larval_MalpighianTubule', 'Larval_Midgut', 'Larval_Carcass'
]
SPECIES_ORDER = ['harm', 'pxyl', 'apis', 'focc', 'drosophila', 'tribolium',
                 'csup', 'silkworm', 'agla', 'ldec', 'nlug', 'lmig']
SP_LABELS = {
    'harm': 'H. armigera', 'pxyl': 'P. xylostella', 'apis': 'A. pisum',
    'focc': 'F. occidentalis', 'drosophila': 'D. melanogaster',
    'tribolium': 'T. castaneum', 'csup': 'C. suppressalis',
    'silkworm': 'B. mori', 'agla': 'A. glabripennis',
    'ldec': 'L. decemlineata', 'nlug': 'N. lugens', 'lmig': 'L. migratoria'
}
TISSUE_SHORT = {t: t.replace('Adult_', 'A.').replace('Larval_', 'L.').replace('MalpighianTubule', 'MalTub') for t in TISSUES}

# ========== load per-species per-tissue Pearson ==========
print('Computing per-species per-tissue Pearson...')
heatmap_data = np.full((len(SPECIES_ORDER), len(TISSUES)), np.nan)
tissue_all_true = {t: [] for t in TISSUES}
tissue_all_pred = {t: [] for t in TISSUES}

for si, sp in enumerate(SPECIES_ORDER):
    f = PRED_DIR / f'{sp}_predictions.csv'
    if not f.exists():
        continue
    df = pd.read_csv(f)
    for ti, t in enumerate(TISSUES):
        tc, tp = f'{t}_true', f'{t}_pred'
        if tc not in df.columns:
            continue
        mask = df[tc].notna() & (df[tc] != '')
        vt = pd.to_numeric(df.loc[mask, tc], errors='coerce')
        vp = pd.to_numeric(df.loc[mask, tp], errors='coerce')
        valid = vt.notna() & vp.notna()
        vt, vp = vt[valid].values, vp[valid].values
        if len(vt) >= 20:
            r, _ = pearsonr(vt, vp)
            heatmap_data[si, ti] = r
            tissue_all_true[t].extend(vt)
            tissue_all_pred[t].extend(vp)

# ========== Figure 4A: tissue x species heatmap ==========
fig, ax = plt.subplots(figsize=(14, 7))
mask = np.isnan(heatmap_data)
sns.heatmap(heatmap_data, ax=ax, cmap='RdYlGn', vmin=0.3, vmax=0.85,
            annot=True, fmt='.3f', mask=mask, linewidths=0.5, linecolor='white',
            xticklabels=[TISSUE_SHORT[t] for t in TISSUES],
            yticklabels=[SP_LABELS[sp] for sp in SPECIES_ORDER],
            cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
# mark missing with X
for si in range(len(SPECIES_ORDER)):
    for ti in range(len(TISSUES)):
        if mask[si, ti]:
            ax.text(ti + 0.5, si + 0.5, 'X', ha='center', va='center', fontsize=8, color='#cccccc')
ax.set_title('(A) Per-species Per-tissue Prediction Performance')
ax.set_xlabel('Tissue')
ax.set_ylabel('Species')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4a_tissue_heatmap.png', bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig4a_tissue_heatmap.pdf', bbox_inches='tight')
plt.close()
print('Fig 4A saved.')

# per-tissue overall Pearson
print('\nPer-tissue overall Pearson:')
tissue_pearsons = {}
for t in TISSUES:
    if len(tissue_all_true[t]) >= 30:
        r, _ = pearsonr(tissue_all_true[t], tissue_all_pred[t])
        tissue_pearsons[t] = r
        print(f'  {t}: n={len(tissue_all_true[t])}, Pearson={r:.4f}')

# ========== Figure 4D: scatter plots for ALL 14 tissues ==========
all_tissues = [t for t in TISSUES if t in tissue_pearsons]
# sort by Pearson descending
all_tissues.sort(key=lambda t: tissue_pearsons[t], reverse=True)

ncols = 4
nrows = (len(all_tissues) + ncols - 1) // ncols  # 14 -> 4 rows
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4*nrows))

for idx, t in enumerate(all_tissues):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]
    true_arr = np.array(tissue_all_true[t])
    pred_arr = np.array(tissue_all_pred[t])
    r = tissue_pearsons[t]
    # subsample if too many points
    if len(true_arr) > 5000:
        sel = np.random.RandomState(42).choice(len(true_arr), 5000, replace=False)
        true_arr, pred_arr = true_arr[sel], pred_arr[sel]
    ax.scatter(true_arr, pred_arr, s=2, alpha=0.12, color='#1f78b4', rasterized=True)
    mn, mx = min(true_arr.min(), pred_arr.min()), max(true_arr.max(), pred_arr.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=0.8, alpha=0.6)
    # fit line
    z = np.polyfit(true_arr, pred_arr, 1)
    xline = np.linspace(mn, mx, 100)
    ax.plot(xline, np.polyval(z, xline), '-', color='#e31a1c', linewidth=1.2, alpha=0.7)
    ax.set_xlabel('Observed log$_2$(TPM+1)', fontsize=9)
    ax.set_ylabel('Predicted log$_2$(TPM+1)', fontsize=9)
    ax.set_title(f'{TISSUE_SHORT[t]}  (r={r:.3f}, n={len(tissue_all_true[t]):,})', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)

# hide extra axes
for idx in range(len(all_tissues), nrows * ncols):
    axes[idx // ncols, idx % ncols].set_visible(False)

fig.suptitle('(D) Predicted vs Observed Expression (All 14 Tissues)', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4d_scatter.png', bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig4d_scatter.pdf', bbox_inches='tight')
plt.close()
print('Fig 4D saved.')
print('All done!')

