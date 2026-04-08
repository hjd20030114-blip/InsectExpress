# Repository Layout

This repository is the English public release for the InsectExpress paper companion package.

## Main directories

- `manuscript/`
  - paper draft used for the release snapshot
- `scripts/paper/`
  - released figure-generation scripts
- `data/processed/`
  - aligned tissue-level processed expression matrices
- `models/`
  - released pretrained checkpoint
- `results/paper/`
  - supplementary tables and exported paper figures
- `results/enformer_v2/`
  - selected summaries from the main model and multi-seed experiments
- `results/external_validation_true/`
  - exported outputs from the released external validation package

## Excluded assets

The following classes of files are intentionally not included in this GitHub release:

- raw sequencing reads
- full genome archives
- heavy intermediate tensors and bundles that exceed practical GitHub-release size
- unrelated lethality-project artifacts
