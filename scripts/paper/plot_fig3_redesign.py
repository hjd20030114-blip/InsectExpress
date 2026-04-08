"""Figure 3 Redesign: Cross-species generalization analysis
Panel A: Connected range dot plot (multi-model per species)
Panel B: Species x Model performance heatmap with delta sidebar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy.stats import pearsonr

RDIR = Path('/home/hjd/RNAi/results/paper')
PDIR = Path('/home/hjd/RNAi/results/predictions_20kb')
FDIR = RDIR / 'figures'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

BL_SP = {0:'tribolium',1:'drosophila',2:'silkworm',3:'pxyl',4:'apis',
    5:'nlug',6:'lmig',7:'harm',8:'csup',9:'focc',10:'agla',11:'ldec'}

# species display info: (code, latin_name, order)
SP_INFO = {
    'harm':       ('H. armigera',       'Lepidoptera'),
    'silkworm':   ('B. mori',           'Lepidoptera'),
    'pxyl':       ('P. xylostella',     'Lepidoptera'),
    'csup':       ('C. suppressalis',   'Lepidoptera'),
    'drosophila': ('D. melanogaster',   'Diptera'),
    'focc':       ('F. occidentalis',   'Thysanoptera'),
    'apis':       ('A. pisum',          'Hemiptera'),
    'nlug':       ('N. lugens',         'Hemiptera'),
    'tribolium':  ('T. castaneum',      'Coleoptera'),
    'agla':       ('A. glabripennis',   'Coleoptera'),
    'ldec':       ('L. decemlineata',   'Coleoptera'),
    'lmig':       ('L. migratoria',     'Orthoptera'),
}

ORDER_COLORS = {
    'Lepidoptera': '#E74C3C', 'Diptera': '#3498DB', 'Thysanoptera': '#2ECC71',
    'Hemiptera': '#9B59B6', 'Coleoptera': '#F39C12', 'Orthoptera': '#1ABC9C',
}

# phylogenetic ordering (by order, within order by InsectExpress performance desc)
PHYLO_ORDER = [
    'Lepidoptera', 'Diptera', 'Thysanoptera', 'Hemiptera', 'Coleoptera', 'Orthoptera'
]

# ---- Load per-species Pearson for all models ----
print('Loading data...')
MODELS = ['InsectExpress', 'MLP', 'XGBoost', 'RF', 'Vanilla Enformer', 'ElasticNet', 'Xpresso']
MODEL_FILES = {
    'MLP': 'baseline_mlp_preds.npz', 'XGBoost': 'baseline_xgboost_preds.npz',
    'RF': 'baseline_rf_preds.npz', 'ElasticNet': 'baseline_elasticnet_preds.npz',
    'Vanilla Enformer': 'baseline_vanilla_enformer_preds.npz',
    'Xpresso': 'baseline_xpresso_preds.npz',
}

# per-species Pearson dict: {model: {species: pearson}}
all_pearsons = {}

# InsectExpress from prediction CSVs
ie_sp = {}
for code in SP_INFO:
    f = PDIR / f'{code}_predictions.csv'
    if not f.exists():
        continue
    df = pd.read_csv(f)
    ts, ps = [], []
    for c in [c for c in df.columns if c.endswith('_true')]:
        pc = c.replace('_true', '_pred')
        if pc in df.columns:
            v = df[[c, pc]].dropna()
            ts.append(v[c].values)
            ps.append(v[pc].values)
    if ts:
        ie_sp[code] = pearsonr(np.concatenate(ts), np.concatenate(ps))[0]
all_pearsons['InsectExpress'] = ie_sp

# baselines
for mname, fname in MODEL_FILES.items():
    d = np.load(RDIR / fname, allow_pickle=True)
    preds, targets, masks, sp_ids = d['preds'], d['targets'], d['masks'], d['species_ids']
    sp_r = {}
    for sid, sp in BL_SP.items():
        idx = sp_ids == sid
        if idx.sum() == 0:
            continue
        m = masks[idx].flatten() > 0.5
        yt, yp = targets[idx].flatten()[m], preds[idx].flatten()[m]
        if len(yt) > 10:
            sp_r[sp] = pearsonr(yt, yp)[0]
    all_pearsons[mname] = sp_r

# build species order: group by order, sort within by InsectExpress desc
species_ordered = []
for order in PHYLO_ORDER:
    sps = [code for code, (_, o) in SP_INFO.items() if o == order]
    sps.sort(key=lambda x: ie_sp.get(x, 0), reverse=True)
    species_ordered.extend(sps)

# reverse for bottom-to-top display
species_ordered = species_ordered[::-1]
n_sp = len(species_ordered)
print(f'{n_sp} species loaded')

# ============================================================
# Panel A: Connected range dot plot
# ============================================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.45)
ax_a = fig.add_subplot(gs[0])

MODEL_MARKERS = {
    'InsectExpress': ('D', '#D62728', 10, 3.0),   # diamond, red
    'MLP': ('o', '#1F77B4', 7, 1.5),
    'XGBoost': ('s', '#2CA02C', 6, 1.0),
    'RF': ('^', '#9467BD', 6, 1.0),
    'Vanilla Enformer': ('P', '#FF7F0E', 7, 1.5),
    'ElasticNet': ('v', '#8C564B', 5, 1.0),
    'Xpresso': ('X', '#AAAAAA', 6, 1.0),
}

# draw order bracket backgrounds
prev_order = None
order_spans = {}
for i, sp in enumerate(species_ordered):
    order = SP_INFO[sp][1]
    if order not in order_spans:
        order_spans[order] = [i, i]
    else:
        order_spans[order][1] = i

for order, (lo, hi) in order_spans.items():
    color = ORDER_COLORS[order]
    ax_a.axhspan(lo - 0.4, hi + 0.4, color=color, alpha=0.06, zorder=0)
    # order label outside axes on far left
    mid = (lo + hi) / 2
    ax_a.annotate(order, xy=(0, mid), xycoords=('axes fraction', 'data'),
                  xytext=(-130, 0), textcoords='offset points',
                  ha='right', va='center', fontsize=9,
                  color=color, fontweight='bold', fontstyle='italic')

for i, sp in enumerate(species_ordered):
    # range line: min to max across all models
    vals = [all_pearsons[m].get(sp, np.nan) for m in MODELS]
    vals_valid = [v for v in vals if not np.isnan(v)]
    if vals_valid:
        ax_a.plot([min(vals_valid), max(vals_valid)], [i, i],
                  color='#DDD', lw=2.5, zorder=1, solid_capstyle='round')

    # plot each model dot
    for mname in MODELS:
        v = all_pearsons[mname].get(sp, np.nan)
        if np.isnan(v):
            continue
        marker, color, ms, lw = MODEL_MARKERS[mname]
        zord = 10 if mname == 'InsectExpress' else 5
        ax_a.scatter(v, i, marker=marker, c=color, s=ms**2, zorder=zord,
                     edgecolors='white' if mname == 'InsectExpress' else 'none',
                     linewidths=1.2 if mname == 'InsectExpress' else 0)

# y labels - smaller font to avoid overlap
ax_a.set_yticks(range(n_sp))
ax_a.set_yticklabels([f'{SP_INFO[sp][0]}' for sp in species_ordered],
                     fontstyle='italic', fontsize=10)
ax_a.set_xlabel('Pearson Correlation', fontweight='bold')
ax_a.set_title('A  Per-species Model Comparison', loc='left', fontweight='bold', fontsize=14)
ax_a.set_xlim(0.25, 0.82)
ax_a.set_ylim(-0.6, n_sp - 0.4)
ax_a.grid(axis='x', alpha=0.15)

# model legend - placed outside plot at bottom right
legend_handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                              markersize=7, label=mname, markeredgecolor='none')
                  for mname, (m, c, _, _) in MODEL_MARKERS.items()]
ax_a.legend(handles=legend_handles, loc='upper center',
            bbox_to_anchor=(0.5, -0.06), fontsize=9,
            framealpha=0.95, ncol=2, title='Model', title_fontsize=10)

# ============================================================
# Panel B: Species x Model heatmap + delta sidebar
# ============================================================
ax_b = fig.add_subplot(gs[1])

# build matrix (species x models)
sel_models = ['InsectExpress', 'MLP', 'Vanilla Enformer', 'XGBoost', 'RF', 'ElasticNet', 'Xpresso']
mat = np.zeros((n_sp, len(sel_models)))
for i, sp in enumerate(species_ordered):
    for j, m in enumerate(sel_models):
        mat[i, j] = all_pearsons[m].get(sp, np.nan)

im = ax_b.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0.25, vmax=0.78,
                 interpolation='nearest')

# annotate values
for i in range(n_sp):
    for j in range(len(sel_models)):
        v = mat[i, j]
        if not np.isnan(v):
            # bold for best in row
            row_max = np.nanmax(mat[i, :])
            fw = 'bold' if abs(v - row_max) < 0.001 else 'normal'
            tc = 'white' if v < 0.4 else '#222'
            ax_b.text(j, i, f'{v:.3f}', ha='center', va='center',
                      fontsize=8.5, fontweight=fw, color=tc)

ax_b.set_xticks(range(len(sel_models)))
ax_b.set_xticklabels(sel_models, rotation=35, ha='right', fontsize=10)
ax_b.set_yticks(range(n_sp))
ax_b.set_yticklabels([f'{SP_INFO[sp][0]}' for sp in species_ordered], fontstyle='italic')

# highlight InsectExpress column
ax_b.axvline(0.5, color='#D62728', lw=0.8, alpha=0.3)

ax_b.set_title('B  Cross-species Performance Heatmap', loc='left', fontweight='bold', fontsize=14)

# colorbar - increased pad to avoid overlap with delta text
cbar = plt.colorbar(im, ax=ax_b, shrink=0.6, pad=0.15)
cbar.set_label('Pearson Correlation', fontsize=11)

# add delta annotations on right side of heatmap (before colorbar)
for i, sp in enumerate(species_ordered):
    ie_v = all_pearsons['InsectExpress'].get(sp, np.nan)
    # best baseline (excl InsectExpress)
    bl_vals = [all_pearsons[m].get(sp, np.nan) for m in sel_models[1:]]
    best_bl = np.nanmax(bl_vals)
    delta = ie_v - best_bl
    ax_b.text(len(sel_models) - 0.3, i, f'+{delta:.3f}', ha='left', va='center',
              fontsize=9, fontweight='bold', color='#D62728' if delta > 0.08 else '#555')

# add "Delta" header
ax_b.text(len(sel_models) - 0.3, -0.8, r'$\Delta$', ha='left', va='center',
          fontsize=10, fontweight='bold', color='#555')

# bbox_inches='tight' handles layout; skip tight_layout to avoid imshow warning
print('Saving fig3_redesign...')
for ext in ['png', 'pdf']:
    fig.savefig(FDIR / f'fig3_redesign.{ext}', dpi=300, bbox_inches='tight')
print(f'Done: {FDIR}/fig3_redesign.png')
plt.close()

