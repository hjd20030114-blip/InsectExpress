# Processed Expression Matrices

This directory contains released aligned expression matrices used in the InsectExpress paper companion package.

## File naming

Each file follows the pattern:

`<species_name>_expression_aligned.tsv`

## Content

Each table stores gene-level expression values aligned to the standardized tissue panel used by InsectExpress.

The standardized tissue panel includes:

- Adult_Brain
- Adult_Head
- Adult_Midgut
- Adult_Hindgut
- Adult_FatBody
- Adult_MalpighianTubule
- Adult_Carcass
- Adult_Ovary
- Adult_Testis
- Larval_Hindgut
- Larval_FatBody
- Larval_MalpighianTubule
- Larval_Midgut
- Larval_Carcass

Missing tissues remain missing in the aligned matrices rather than being converted to artificial zero values.

## Release scope

These processed matrices are released for figure inspection, downstream validation, and repository-level reproducibility of the exported paper assets. This GitHub package does not include the full raw read-processing workflow or raw sequencing inputs.
