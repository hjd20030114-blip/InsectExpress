"""Supplementary figures:
S2: Expression-level stratified Pearson (InsectExpress vs baselines)
S3: Tissue-specific vs housekeeping gene prediction accuracy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

RDIR = Path('/home/hjd/RNAi/results/paper')
PDIR = Path('/home/hjd/RNAi/results/predictions_20kb')
FDIR = RDIR / 'figures'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14,
    'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_OURS, C_BEST, C_VE, C_GRAY = '#D62728', '#1F77B4', '#2CA02C', '#AAAAAA'

BL_SP = {0:'tribolium',1:'drosophila',2:'silkworm',3:'pxyl',4:'apis',
    5:'nlug',6:'lmig',7:'harm',8:'csup',9:'focc',10:'agla',11:'ldec'}

TISSUES = [
    'Adult_Brain', 'Adult_Head', 'Adult_Midgut', 'Adult_Hindgut',
    'Adult_FatBody', 'Adult_MalpighianTubule', 'Adult_Carcass',
    'Adult_Ovary', 'Adult_Testis',
    'Larval_Hindgut', 'Larval_FatBody', 'Larval_MalpighianTubule',
    'Larval_Midgut', 'Larval_Carcass',
]

def fv(p, t, m):
    mk = m.flatten() > 0.5
    return t.flatten()[mk], p.flatten()[mk]

# ---- Load data ----
print('Loading baselines...')
bl_raw = {}
for name, fname in [('MLP','baseline_mlp_preds.npz'),
                     ('Vanilla Enformer','baseline_vanilla_enformer_preds.npz'),
                     ('Xpresso','baseline_xpresso_preds.npz')]:
    d = np.load(RDIR / fname, allow_pickle=True)
    yt, yp = fv(d['preds'], d['targets'], d['masks'])
    bl_raw[name] = (yt, yp)

print('Loading InsectExpress predictions...')
ie_trues, ie_preds = [], []
gene_tissue_data = []  # for tissue-specificity analysis
for sp in BL_SP.values():
    f = PDIR / f'{sp}_predictions.csv'
    if not f.exists():
        continue
    df = pd.read_csv(f)
    for c in [c for c in df.columns if c.endswith('_true')]:
        pc = c.replace('_true', '_pred')
        if pc in df.columns:
            v = df[[c, pc]].dropna()
            ie_trues.append(v[c].values)
            ie_preds.append(v[pc].values)
    # per-gene tissue profile for specificity analysis
    true_cols = [f'{t}_true' for t in TISSUES if f'{t}_true' in df.columns]
    pred_cols = [f'{t}_pred' for t in TISSUES if f'{t}_pred' in df.columns]
    if true_cols:
        tvals = df[true_cols].values  # (n_genes, n_tissues)
        pvals = df[pred_cols].values
        for i in range(len(df)):
            row_t = tvals[i]
            row_p = pvals[i]
            valid = ~np.isnan(row_t) & ~np.isnan(row_p)
            if valid.sum() >= 3:
                gene_tissue_data.append((row_t[valid], row_p[valid], sp))

ie_yt = np.concatenate(ie_trues)
ie_yp = np.concatenate(ie_preds)
bl_raw['InsectExpress'] = (ie_yt, ie_yp)

# ============================================================
# Figure S2: Expression-level stratified Pearson
# ============================================================
print('Computing stratified Pearson...')
strata = [
    ('Silent\n(0)', 0, 0.01),
    ('Very Low\n(0-1)', 0.01, 1.0),
    ('Low\n(1-3)', 1.0, 3.0),
    ('Medium\n(3-6)', 3.0, 6.0),
    ('High\n(6-10)', 6.0, 10.0),
    ('Very High\n(>10)', 10.0, 20.0),
]

fig, ax = plt.subplots(figsize=(10, 6))
models = [('InsectExpress', C_OURS), ('MLP', C_BEST),
          ('Vanilla Enformer', C_VE), ('Xpresso', C_GRAY)]
x = np.arange(len(strata))
w = 0.18

for mi, (mname, mc) in enumerate(models):
    yt, yp = bl_raw[mname]
    vals, counts = [], []
    for sname, lo, hi in strata:
        mask = (yt >= lo) & (yt < hi)
        cnt = mask.sum()
        counts.append(cnt)
        if cnt >= 30:
            vals.append(pearsonr(yt[mask], yp[mask])[0])
        else:
            vals.append(np.nan)
    offset = (mi - 1.5) * w
    bars = ax.bar(x + offset, vals, w * 0.9, color=mc, alpha=0.85, label=mname)
    # count labels on InsectExpress bars only
    if mname == 'InsectExpress':
        for xi, (v, cnt) in enumerate(zip(vals, counts)):
            if not np.isnan(v):
                lbl = f'{cnt//1000}k' if cnt >= 1000 else str(cnt)
                ax.text(xi + offset, v + 0.01, lbl, ha='center', fontsize=8, color='#666')

ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in strata])
ax.set_xlabel('Expression Level (log\u2082(TPM+1))')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Prediction Accuracy by Expression Level', fontweight='bold')
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.15)
ax.set_ylim(-0.05, 0.85)

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(FDIR / f'figS2_stratified_expression.{ext}', dpi=300, bbox_inches='tight')
print(f'Done: figS2_stratified_expression.png')
plt.close()

# ============================================================
# Figure S3: Tissue-specific vs housekeeping gene analysis
# ============================================================
print('Analyzing tissue specificity...')
spec_scores, spec_pearsons = [], []
for true_profile, pred_profile, sp in gene_tissue_data:
    mean_expr = np.mean(true_profile)
    if mean_expr < 0.1:
        continue
    max_expr = np.max(true_profile)
    # tau specificity index
    n = len(true_profile)
    if max_expr > 0:
        tau = np.sum(1 - true_profile / max_expr) / (n - 1) if n > 1 else 0
    else:
        tau = 0
    # per-gene correlation across tissues
    if len(true_profile) >= 3 and np.std(true_profile) > 0.01 and np.std(pred_profile) > 0.01:
        r = pearsonr(true_profile, pred_profile)[0]
        spec_scores.append(tau)
        spec_pearsons.append(r)

spec_scores = np.array(spec_scores)
spec_pearsons = np.array(spec_pearsons)
print(f'  {len(spec_scores)} genes with valid tau + Pearson')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Tau distribution colored by prediction accuracy
sc = ax1.scatter(spec_scores, spec_pearsons, s=3, alpha=0.15, c=spec_pearsons,
                 cmap='RdYlGn', vmin=-0.2, vmax=1.0, rasterized=True)
# bin means
tau_bins = np.linspace(0, 1, 11)
for i in range(len(tau_bins) - 1):
    mask = (spec_scores >= tau_bins[i]) & (spec_scores < tau_bins[i+1])
    if mask.sum() >= 20:
        bc = (tau_bins[i] + tau_bins[i+1]) / 2
        ax1.scatter(bc, np.median(spec_pearsons[mask]), s=80, c=C_OURS,
                    edgecolors='white', linewidths=1.5, zorder=10, marker='D')

ax1.set_xlabel('Tissue Specificity Index (Tau)')
ax1.set_ylabel('Per-gene Pearson (across tissues)')
ax1.set_title('A  Tissue Specificity vs Prediction Accuracy', loc='left', fontweight='bold')
ax1.axhline(0, color='#ccc', ls='--', lw=1)
plt.colorbar(sc, ax=ax1, shrink=0.7, label='Per-gene Pearson')

# Panel B: Boxplot by tau category
cats = [('Housekeeping\n(tau<0.3)', spec_scores < 0.3),
        ('Moderate\n(0.3-0.7)', (spec_scores >= 0.3) & (spec_scores < 0.7)),
        ('Specific\n(tau>0.7)', spec_scores >= 0.7)]
bp_data = [spec_pearsons[m] for _, m in cats]
colors_bp = ['#3498DB', '#F39C12', '#E74C3C']
bp = ax2.boxplot(bp_data, patch_artist=True, widths=0.6, showfliers=False,
                 medianprops={'color': C_OURS, 'lw': 2})
for patch, c in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(c)
    patch.set_alpha(0.3)
for i, (label, mask) in enumerate(cats):
    n = mask.sum()
    med = np.median(spec_pearsons[mask])
    ax2.text(i + 1, med + 0.03, f'n={n}\nmed={med:.3f}', ha='center', fontsize=10, fontweight='bold')

ax2.set_xticks(np.arange(1, len(cats) + 1))
ax2.set_xticklabels([c[0] for c in cats])
ax2.set_ylabel('Per-gene Pearson (across tissues)')
ax2.set_title('B  Prediction by Gene Category', loc='left', fontweight='bold')
ax2.grid(axis='y', alpha=0.15)

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(FDIR / f'figS3_tissue_specificity.{ext}', dpi=300, bbox_inches='tight')
print(f'Done: figS3_tissue_specificity.png')
plt.close()
