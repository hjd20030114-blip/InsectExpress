#!/usr/bin/env python3
"""
Figure 1A,B: Species data overview
  Panel A: Species phylogeny cladogram (text-based, for BioRender reference)
  Panel B: Species x Tissue coverage heatmap (12 species x 14 tissues)

Output: results/paper/figures/fig1_data_overview.pdf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training'))

TISSUE_NAMES = [
    'Adult_Brain', 'Adult_Head', 'Adult_Midgut', 'Adult_Hindgut',
    'Adult_FatBody', 'Adult_MalpighianTubule', 'Adult_Carcass',
    'Adult_Ovary', 'Adult_Testis',
    'Larval_Hindgut', 'Larval_FatBody', 'Larval_MalpighianTubule',
    'Larval_Midgut', 'Larval_Carcass'
]

# phylogenetic order (by taxonomic relationship)
SPECIES_INFO = [
    # (file_prefix, display_name, order, order_color)
    ('drosophila',               'D. melanogaster',    'Diptera',       '#e74c3c'),
    ('tribolium',                'T. castaneum',       'Coleoptera',    '#3498db'),
    ('anoplophora_glabripennis', 'A. glabripennis',    'Coleoptera',    '#3498db'),
    ('leptinotarsa_decemlineata','L. decemlineata',    'Coleoptera',    '#3498db'),
    ('silkworm',                 'B. mori',            'Lepidoptera',   '#2ecc71'),
    ('plutella_xylostella',      'P. xylostella',      'Lepidoptera',   '#2ecc71'),
    ('helicoverpa_armigera',     'H. armigera',        'Lepidoptera',   '#2ecc71'),
    ('chilo_suppressalis',       'C. suppressalis',    'Lepidoptera',   '#2ecc71'),
    ('acyrthosiphon_pisum',      'A. pisum',           'Hemiptera',     '#9b59b6'),
    ('nilaparvata_lugens',       'N. lugens',          'Hemiptera',     '#9b59b6'),
    ('locusta_migratoria',       'L. migratoria',      'Orthoptera',    '#f39c12'),
    ('frankliniella_occidentalis','F. occidentalis',   'Thysanoptera',  '#1abc9c'),
]

# per-species per-tissue RNA-seq sample/profile counts
SAMPLE_COUNTS = {
    # model organisms: counts from database profiles (FlyAtlas2/BeetleAtlas/SilkDB)
    # only list tissues that have actual non-NaN data in expression_aligned.tsv
    'drosophila': {
        'Adult_Brain': 2, 'Adult_Head': 2, 'Adult_Midgut': 2, 'Adult_Hindgut': 2,
        'Adult_FatBody': 2, 'Adult_MalpighianTubule': 2, 'Adult_Carcass': 2,
        'Adult_Ovary': 1, 'Adult_Testis': 1,
        'Larval_FatBody': 1, 'Larval_Midgut': 1, 'Larval_Carcass': 1,
    },
    'tribolium': {
        'Adult_Brain': 1, 'Adult_Head': 1, 'Adult_Midgut': 2, 'Adult_Hindgut': 1,
        'Adult_FatBody': 1, 'Adult_MalpighianTubule': 1, 'Adult_Carcass': 1,
        'Larval_Hindgut': 1, 'Larval_FatBody': 1,
        'Larval_Midgut': 2, 'Larval_Carcass': 1,
    },
    'silkworm': {
        'Adult_Head': 1, 'Adult_Midgut': 1, 'Adult_FatBody': 1,
        'Adult_MalpighianTubule': 1, 'Adult_Carcass': 1,
        'Adult_Ovary': 1, 'Adult_Testis': 1,
        'Larval_FatBody': 1, 'Larval_MalpighianTubule': 1,
        'Larval_Midgut': 1, 'Larval_Carcass': 1,
    },
    # pest species: SRR run counts from Table S2
    'anoplophora_glabripennis': {
        'Adult_Brain': 32, 'Adult_FatBody': 53, 'Adult_Hindgut': 15,
        'Adult_MalpighianTubule': 15, 'Adult_Midgut': 25,
    },
    'leptinotarsa_decemlineata': {
        'Adult_Carcass': 4, 'Adult_FatBody': 15, 'Adult_Head': 6,
        'Adult_Hindgut': 3, 'Adult_Ovary': 3, 'Adult_Testis': 3,
        'Larval_Hindgut': 2,
    },
    'plutella_xylostella': {
        'Adult_Brain': 1, 'Adult_Testis': 6,
        'Larval_Carcass': 6, 'Larval_Midgut': 19,
    },
    'helicoverpa_armigera': {
        'Adult_Brain': 22, 'Adult_Carcass': 16, 'Adult_FatBody': 48,
        'Adult_Head': 32, 'Adult_MalpighianTubule': 5,
        'Adult_Midgut': 148, 'Adult_Ovary': 1,
    },
    'chilo_suppressalis': {
        'Adult_Carcass': 12, 'Adult_FatBody': 1, 'Adult_Head': 2,
        'Adult_Midgut': 7, 'Larval_Midgut': 1,
    },
    'acyrthosiphon_pisum': {
        'Adult_Carcass': 1, 'Adult_FatBody': 2, 'Adult_Midgut': 1,
    },
    'nilaparvata_lugens': {
        'Adult_Brain': 9, 'Adult_FatBody': 9, 'Adult_Ovary': 9,
    },
    'locusta_migratoria': {
        'Adult_Brain': 40, 'Adult_Carcass': 2,
        'Adult_Head': 2, 'Adult_Ovary': 3, 'Adult_Testis': 7,
    },
    'frankliniella_occidentalis': {
        'Adult_Carcass': 8, 'Adult_Head': 6,
        'Larval_Carcass': 1, 'Larval_Midgut': 12,
    },
}

# per-species RNA-seq raw data size (GB)
# pest species: fastq.gz total; model species: database file size
# harm estimated from avg per-SRR (131/282 fastq cleaned after processing)
DATA_SIZE_GB = {
    'drosophila': 1.0,
    'tribolium': 0.7,
    'anoplophora_glabripennis': 983,
    'leptinotarsa_decemlineata': 205,
    'silkworm': 0.6,
    'plutella_xylostella': 223,
    'helicoverpa_armigera': 1981,
    'chilo_suppressalis': 191,
    'acyrthosiphon_pisum': 40,
    'nilaparvata_lugens': 174,
    'locusta_migratoria': 777,
    'frankliniella_occidentalis': 288,
}

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUT_DIR = PROJECT_ROOT / 'results' / 'paper' / 'figures'


def load_coverage_matrix():
    """build 12x14 coverage matrix: 1=has data, 0=missing"""
    n_sp = len(SPECIES_INFO)
    n_ts = len(TISSUE_NAMES)
    coverage = np.zeros((n_sp, n_ts))
    gene_counts = []

    for i, (prefix, _, _, _) in enumerate(SPECIES_INFO):
        fpath = DATA_DIR / f'{prefix}_expression_aligned.tsv'
        df = pd.read_csv(fpath, sep='\t', nrows=5)
        n_genes = sum(1 for _ in open(fpath)) - 1
        gene_counts.append(n_genes)
        for j, tissue in enumerate(TISSUE_NAMES):
            if tissue in df.columns:
                # check if any non-NaN values exist
                sample = pd.read_csv(fpath, sep='\t', usecols=['gene_id', tissue])
                if sample[tissue].notna().sum() > 0:
                    coverage[i, j] = 1
    return coverage, gene_counts


def plot_heatmap(coverage, gene_counts):
    """Panel B: species x tissue heatmap"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={'width_ratios': [1, 6], 'wspace': 0.02})

    # left panel: gene count bar chart
    ax_bar = axes[0]
    species_labels = [info[1] for info in SPECIES_INFO]
    colors = [info[3] for info in SPECIES_INFO]
    y_pos = np.arange(len(species_labels))

    ax_bar.barh(y_pos, gene_counts, color=colors, edgecolor='white', height=0.7)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f'$\\it{{{s}}}$' for s in species_labels], fontsize=9)
    ax_bar.set_xlabel('# Genes', fontsize=10)
    ax_bar.set_xlim(0, max(gene_counts) * 1.15)
    ax_bar.invert_yaxis()
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    for i, v in enumerate(gene_counts):
        ax_bar.text(v + 200, i, f'{v:,}', va='center', fontsize=7, color='#555')

    # right panel: coverage heatmap
    ax_hm = axes[1]
    cmap = ListedColormap(['#f0f0f0', '#2980b9'])
    ax_hm.imshow(coverage, cmap=cmap, aspect='auto', interpolation='nearest')

    # tissue labels on top
    tissue_short = [t.replace('Adult_', 'A:').replace('Larval_', 'L:')
                    for t in TISSUE_NAMES]
    ax_hm.set_xticks(np.arange(len(TISSUE_NAMES)))
    ax_hm.set_xticklabels(tissue_short, rotation=55, ha='left', fontsize=8)
    ax_hm.xaxis.tick_top()
    ax_hm.set_yticks([])

    # grid lines
    for i in range(len(SPECIES_INFO) + 1):
        ax_hm.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(len(TISSUE_NAMES) + 1):
        ax_hm.axvline(j - 0.5, color='white', linewidth=1.5)


    # tissue count + data size annotation per species
    for i, (prefix, _, _, _) in enumerate(SPECIES_INFO):
        n = int(coverage[i].sum())
        size_gb = DATA_SIZE_GB.get(prefix, 0)
        if size_gb >= 1000:
            size_str = f'{size_gb/1024:.1f} TB'
        elif size_gb >= 1:
            size_str = f'{size_gb:.0f} GB'
        else:
            size_str = f'{size_gb*1024:.0f} MB'
        ax_hm.text(len(TISSUE_NAMES) + 0.3, i, f'{n}/14 | {size_str}', va='center',
                   fontsize=7, color='#333', fontweight='bold')

    # title
    total_genes = sum(gene_counts)
    total_entries = int(coverage.sum())
    total_data = sum(DATA_SIZE_GB.get(info[0], 0) for info in SPECIES_INFO)
    fig.suptitle(f'InsectExpress: 12 species, 14 tissues, {total_genes:,} genes, '
                 f'{total_entries} species-tissue combinations, ~{total_data/1024:.1f} TB RNA-seq data',
                 fontsize=12, fontweight='bold', y=0.02)

    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading coverage data...')
    coverage, gene_counts = load_coverage_matrix()
    print(f'Coverage matrix: {coverage.shape}, total entries: {int(coverage.sum())}')

    print('Plotting heatmap...')
    fig = plot_heatmap(coverage, gene_counts)
    out_pdf = OUT_DIR / 'fig1b_data_overview.pdf'
    out_png = OUT_DIR / 'fig1b_data_overview.png'
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()

