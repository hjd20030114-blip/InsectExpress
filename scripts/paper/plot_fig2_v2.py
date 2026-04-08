"""Figure 2 (v2): Benchmarking InsectExpress performance
A: Forest plot + bootstrap CI  B: Species dumbbell plot
C: Multi-metric heatmap  D: Expression abundance error trend
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RDIR = Path('/home/hjd/RNAi/results/paper')
PDIR = Path('/home/hjd/RNAi/results/predictions_20kb')
FDIR = RDIR / 'figures'
FDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
})
C_OURS, C_BEST, C_GRAY = '#D62728', '#1F77B4', '#AAAAAA'

# baseline species ID mapping (from ML baselines log order)
BL_SP = {0:'tribolium',1:'drosophila',2:'silkworm',3:'pxyl',4:'apis',
    5:'nlug',6:'lmig',7:'harm',8:'csup',9:'focc',10:'agla',11:'ldec'}

BL_FILES = [
    ('MLP', 'baseline_mlp_preds.npz'),
    ('XGBoost', 'baseline_xgboost_preds.npz'),
    ('RF', 'baseline_rf_preds.npz'),
    ('Vanilla Enformer', 'baseline_vanilla_enformer_preds.npz'),
    ('ElasticNet', 'baseline_elasticnet_preds.npz'),
    ('MLP-RNA', 'baseline_mlp-rna_preds.npz'),
    ('Xpresso', 'baseline_xpresso_preds.npz'),
]

SP_DISPLAY = {'tribolium':'T. castaneum','drosophila':'D. melanogaster',
    'silkworm':'B. mori','pxyl':'P. xylostella','apis':'A. pisum',
    'nlug':'N. lugens','lmig':'L. migratoria','harm':'H. armigera',
    'csup':'C. suppressalis','focc':'F. occidentalis','agla':'A. glabripennis',
    'ldec':'L. decemlineata'}


def fv(p, t, m):
    mk = m.flatten() > 0.5
    return t.flatten()[mk], p.flatten()[mk]

def calc_m(yt, yp):
    return {'Pearson': pearsonr(yt, yp)[0], 'Spearman': spearmanr(yt, yp)[0],
            'R2': r2_score(yt, yp), 'RMSE': np.sqrt(mean_squared_error(yt, yp)),
            'MAE': mean_absolute_error(yt, yp)}

def bci(yt, yp, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    rs = []
    for _ in range(n):
        idx = rng.choice(len(yt), len(yt), True)
        rs.append(pearsonr(yt[idx], yp[idx])[0])
    return np.percentile(rs, [2.5, 97.5])

def per_sp(p, t, m, sid):
    res = {}
    for s in np.unique(sid):
        yt, yp = fv(p[sid == s], t[sid == s], m[sid == s])
        if len(yt) >= 10:
            res[BL_SP[int(s)]] = pearsonr(yt, yp)[0]
    return res


# ---- Load data ----
print('Loading baselines...')
M, CI, SP = {}, {}, {}
bl_raw = {}
for name, fname in BL_FILES:
    d = np.load(RDIR / fname, allow_pickle=True)
    yt, yp = fv(d['preds'], d['targets'], d['masks'])
    M[name] = calc_m(yt, yp)
    CI[name] = bci(yt, yp)
    SP[name] = per_sp(d['preds'], d['targets'], d['masks'], d['species_ids'])
    bl_raw[name] = (yt, yp)
    print(f'  {name}: Pearson={M[name]["Pearson"]:.4f}')

print('Loading InsectExpress...')
ie_t, ie_p, ie_sp = [], [], {}
for sp in BL_SP.values():
    f = PDIR / f'{sp}_predictions.csv'
    if not f.exists():
        continue
    df = pd.read_csv(f)
    st, sp_p = [], []
    for c in [c for c in df.columns if c.endswith('_true')]:
        pc = c.replace('_true', '_pred')
        if pc in df.columns:
            v = df[[c, pc]].dropna()
            st.append(v[c].values)
            sp_p.append(v[pc].values)
    if st:
        a, b = np.concatenate(st), np.concatenate(sp_p)
        ie_sp[sp] = pearsonr(a, b)[0]
        ie_t.append(a)
        ie_p.append(b)

ie_yt, ie_yp = np.concatenate(ie_t), np.concatenate(ie_p)
M['InsectExpress'] = calc_m(ie_yt, ie_yp)
CI['InsectExpress'] = bci(ie_yt, ie_yp)
SP['InsectExpress'] = ie_sp
print(f'  InsectExpress: Pearson={M["InsectExpress"]["Pearson"]:.4f}')

# sort by Pearson
order = sorted(M.keys(), key=lambda x: M[x]['Pearson'])

# ---- Figure ----
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.4)

# Panel A: Forest plot
ax = fig.add_subplot(gs[0, 0])
for i, nm in enumerate(order):
    pr = M[nm]['Pearson']
    ci = CI[nm]
    c = C_OURS if nm == 'InsectExpress' else (C_BEST if nm == 'MLP' else C_GRAY)
    z = 10 if nm == 'InsectExpress' else 5
    ax.errorbar(pr, i, xerr=[[pr - ci[0]], [ci[1] - pr]], fmt='o',
                color=c, ms=8, capsize=4, capthick=1.5, elinewidth=1.5, zorder=z)
    fw = 'bold' if nm == 'InsectExpress' else 'normal'
    ax.text(ci[1] + 0.005, i, f'{pr:.3f}', va='center', fontsize=11, fontweight=fw, color=c)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xlabel('Pearson Correlation')
ax.set_title('A  Overall Model Comparison', loc='left', fontweight='bold')
ax.axvline(M['InsectExpress']['Pearson'], color=C_OURS, alpha=0.15, ls='--', zorder=0)

# Panel B: Dumbbell plot
ax = fig.add_subplot(gs[0, 1])
best_bl = 'MLP'
sp_order = sorted(ie_sp.keys(), key=lambda x: ie_sp[x])
for i, sp in enumerate(sp_order):
    ie_v = ie_sp.get(sp, np.nan)
    bl_v = SP[best_bl].get(sp, np.nan)
    if np.isnan(bl_v):
        continue
    ax.plot([bl_v, ie_v], [i, i], '-', color='#DDDDDD', lw=2.5, zorder=1)
    ax.scatter(bl_v, i, color=C_BEST, s=60, zorder=5, edgecolors='white', linewidths=0.5)
    ax.scatter(ie_v, i, color=C_OURS, s=60, zorder=5, edgecolors='white', linewidths=0.5)
    d = ie_v - bl_v
    ax.text(max(ie_v, bl_v) + 0.008, i, f'+{d:.3f}', va='center', fontsize=10, color='#333')
ax.set_yticks(range(len(sp_order)))
ax.set_yticklabels([SP_DISPLAY.get(s, s) for s in sp_order], fontstyle='italic')
ax.set_xlabel('Pearson Correlation')
ax.set_title('B  Species-level Improvement over Best Overall Baseline (MLP)', loc='left', fontweight='bold')


# Panel C: Multi-metric heatmap
ax = fig.add_subplot(gs[1, 0])
metrics = ['Pearson', 'Spearman', 'R2', 'RMSE', 'MAE']
hm_models = list(reversed(order))
hm_data = pd.DataFrame({nm: M[nm] for nm in hm_models}).T[metrics]
# for display: higher is better for Pearson/Spearman/R2, lower for RMSE/MAE
# use rank-based coloring
rank_df = hm_data.copy()
for c in ['Pearson', 'Spearman', 'R2']:
    rank_df[c] = hm_data[c].rank()
for c in ['RMSE', 'MAE']:
    rank_df[c] = (-hm_data[c]).rank()  # invert: lower is better
annot = hm_data.copy()
for c in metrics:
    annot[c] = hm_data[c].map(lambda x: f'{x:.3f}')
sns.heatmap(rank_df, annot=annot.values, fmt='', cmap='RdYlGn', ax=ax,
            linewidths=0.8, linecolor='white',
            cbar_kws={'label': 'Column-wise rank (higher = better)', 'shrink': 0.7},
            xticklabels=['Pearson', 'Spearman', 'R\u00b2', 'RMSE\u2193', 'MAE\u2193'])
# highlight InsectExpress row: light fill + thick border
ie_idx = hm_models.index('InsectExpress')
ax.add_patch(plt.Rectangle((0, ie_idx), len(metrics), 1, fill=True,
             facecolor=C_OURS, alpha=0.08, edgecolor=C_OURS, lw=3.0))
ax.set_title('C  Multi-metric Comparison', loc='left', fontweight='bold')
ax.set_ylabel('')

# Panel D: Expression abundance error trend (binned line + ribbon)
ax = fig.add_subplot(gs[1, 1])
# use InsectExpress data, bin by true expression level
bins = np.arange(0, 14, 1.0)
bin_centers = (bins[:-1] + bins[1:]) / 2
# models to compare
dsets = {'InsectExpress': (ie_yt, ie_yp)}
for nm in ['MLP', 'Vanilla Enformer', 'Xpresso']:
    if nm in bl_raw:
        dsets[nm] = bl_raw[nm]

colors = {'InsectExpress': C_OURS, 'MLP': C_BEST, 'Vanilla Enformer': '#2CA02C', 'Xpresso': C_GRAY}

# pre-compute per-bin counts (from InsectExpress data, shared x-axis)
bin_counts = []
for b in range(len(bins) - 1):
    bin_counts.append(((ie_yt >= bins[b]) & (ie_yt < bins[b + 1])).sum())

for nm, (yt, yp) in dsets.items():
    ae = np.abs(yt - yp)
    digitized = np.digitize(yt, bins) - 1
    medians, lo, hi, xs = [], [], [], []
    for b in range(len(bins) - 1):
        mask = digitized == b
        if mask.sum() < 30:
            continue
        vals = ae[mask]
        medians.append(np.median(vals))
        # bootstrap 90% CI (wider band for visibility)
        rng = np.random.RandomState(42)
        bs = [np.median(rng.choice(vals, len(vals), True)) for _ in range(500)]
        lo.append(np.percentile(bs, 5))
        hi.append(np.percentile(bs, 95))
        xs.append(bin_centers[b])
    ax.plot(xs, medians, '-o', color=colors.get(nm, C_GRAY), ms=5, lw=2.0, label=nm, zorder=5)
    ax.fill_between(xs, lo, hi, alpha=0.28, color=colors.get(nm, C_GRAY), zorder=3)

# count bar on twin axis (prominent)
ax2 = ax.twinx()
ax2.bar(bin_centers, bin_counts, width=0.85, alpha=0.18, color='#555555', zorder=0)
ax2.set_ylabel('Gene count per bin', color='#555', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#555', labelsize=11)
ax2.spines['top'].set_visible(False)
# annotate counts on bars
for bc, cnt in zip(bin_centers, bin_counts):
    if cnt > 0:
        lbl = f'{cnt/1000:.0f}k' if cnt >= 1000 else str(cnt)
        ax2.text(bc, cnt, lbl, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#666')

ax.set_xlabel('True Expression Level (log\u2082(TPM+1))')
ax.set_ylabel('Median Absolute Error')
ax.set_title('D  Performance across Expression Abundance', loc='left', fontweight='bold')


print('Saving...')
for ext in ['png', 'pdf']:
    fig.savefig(FDIR / f'fig2_v2_benchmark.{ext}', dpi=300, bbox_inches='tight')
print(f'Done: {FDIR}/fig2_v2_benchmark.png')
plt.close()

