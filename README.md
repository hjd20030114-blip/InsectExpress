# InsectExpress

InsectExpress is a multimodal framework for cross-species prediction of tissue-resolved gene expression in insects.

This repository is the public English companion release for the InsectExpress manuscript. It contains:

- the manuscript draft used for the paper submission package
- processed aligned expression matrices used in the released analyses
- the main model checkpoint
- figure-generation scripts used for the paper figures
- exported figures, supplementary tables, and validation summaries

## Repository layout

- `manuscript/`: manuscript files
- `scripts/paper/`: figure-generation scripts used in the paper companion release
- `data/processed/`: released processed expression matrices in aligned tissue space
- `models/`: released pretrained checkpoint
- `results/paper/`: main paper tables and figures
- `results/enformer_v2/`: selected model summaries and multi-seed outputs
- `results/external_validation_true/`: released external-validation outputs
- `docs/`: repository notes and data descriptions

## Included assets

- Main manuscript: `manuscript/InsectExpress_manuscript.docx`
- Main checkpoint: `models/insectexpress_seed42_checkpoint.pt`
- Supplementary tables: `results/paper/Table_S1_species_genome_info.csv` to `Table_S4_model_hyperparameters.csv`
- Main and supplementary figures: `results/paper/figures/`
- Multi-seed summary and figures: `results/enformer_v2/`
- External validation prediction exports and audit summaries: `results/external_validation_true/`

## Scope of this release

This GitHub repository is a processed-data and paper-assets release. It does not include the full raw-data acquisition pipeline, raw genome archives, raw RNA-seq reads, or the largest intermediate tensors generated during interpretability analysis.

Large internal files intentionally excluded from this GitHub release include:

- raw sequencing inputs and genome assemblies
- precomputed internal attention tensors larger than the GitHub file-size limit
- heavy intermediate bundles not required to inspect the released figures

## Software environment

The paper figures and released scripts were prepared in the following environment:

- Python 3.10.12
- PyTorch 2.7.1+cu118
- scikit-learn 1.7.2
- XGBoost 3.1.1
- NumPy 1.26.4
- pandas 2.3.3
- matplotlib 3.10.7
- seaborn 0.13.2

See `requirements.txt` for the pinned package versions used in this release snapshot.

## Reproducing released figures

The scripts in `scripts/paper/` are intended for regeneration of the released paper figures from the exported processed tables and result files in this repository. Most scripts write outputs into `results/paper/figures/` relative to the repository root.

Example:

```bash
python scripts/paper/plot_fig6_ism_interpretability.py
python scripts/paper/plot_fig_external_validation_true.py
```

## Notes

- All public file and directory names in this repository use English naming.
- This repository release is focused on the InsectExpress expression-prediction project and excludes unrelated lethality-project assets.
