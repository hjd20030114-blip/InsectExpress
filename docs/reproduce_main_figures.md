# Reproducing the Main Figures

This document summarizes how to regenerate the released paper figures from the
processed files included in this repository.

## Prerequisites

Install the Python environment listed in `requirements.txt`.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are expected to be run from the repository root.

## General note

The public repository is a paper-companion release. The figure scripts are
designed to regenerate exported figures from the released processed tables and
summary files already included here. This package is not a full raw-data
reprocessing release.

## Main figure commands

### Figure 1

```bash
python scripts/paper/plot_fig1_data_overview.py
python scripts/paper/plot_fig1c_architecture.py
```

Expected outputs in `results/paper/figures/`:

- `fig1b_data_overview.png`
- `fig1b_data_overview.pdf`
- `fig1c_architecture.png`
- `fig1c_architecture.pdf`

### Figure 2

```bash
python scripts/paper/plot_fig2_v2.py
```

Expected outputs include:

- `fig2_v2_benchmark.png`
- `fig2a_overall_comparison.png`
- `fig2b_stratified.png`
- `fig2c_radar.png`
- `fig2d_convergence.png`

### Figure 3

```bash
python scripts/paper/plot_fig3_redesign.py
```

Expected outputs include:

- `fig3_redesign.png`
- `fig3a_per_species.png`
- `fig3b_phylo_performance.png`

### Figure 4

```bash
python scripts/paper/plot_fig4_tissue.py
python scripts/paper/plot_fig4bc_tissue.py
```

Expected outputs include:

- `fig4a_tissue_heatmap.png`
- `fig4bc_tissue_analysis.png`
- `fig4d_scatter.png`

### Figure 5

Figure 5 assets are already released in `results/paper/figures/`. The released
public package focuses on the paper-facing output tables and exported figures.

### Figure 6

```bash
python scripts/paper/plot_fig6_ism_interpretability.py
```

Expected outputs include:

- `fig6_ism_interpretability.png`
- `fig6_ism_shared_motifs.csv`
- `fig6_ism_signature_motifs.csv`

### Figure 7

```bash
python scripts/paper/plot_fig7_attention.py
```

Expected outputs include:

- `fig7_attention.png`
- `fig7_candidate_gene_stats.csv`
- `fig7_selected_genes.csv`

### External validation figure

```bash
python scripts/paper/plot_fig_external_validation_true.py
```

Expected outputs include:

- `fig_external_validation_true.png`
- `fig_external_validation_true_metrics.csv`
- `fig_external_validation_true_per_tissue.csv`
- `fig_external_validation_true_audit.csv`

## Supplementary figures

```bash
python scripts/paper/plot_figS_convergence.py
python scripts/paper/plot_figS23_supp.py
python scripts/paper/plot_fig_ablation.py
```

Expected outputs include:

- `figS_convergence.png`
- `figS2_stratified_expression.png`
- `figS3_tissue_specificity.png`
- `fig_ablation.png`

## Output location

Unless a script overrides the path internally, released figure outputs are
written into:

```text
results/paper/figures/
```

## Troubleshooting

- If a script fails because of a missing file, first confirm that the required
  processed tables and released summaries are present under `results/` and
  `data/processed/`.
- Some scripts assume the repository root as the current working directory.
- The public package does not include all internal training intermediates, so
  figure regeneration is supported for the released paper-facing assets rather
  than every internal exploratory artifact.
