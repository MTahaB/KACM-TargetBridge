# Optional Structure-Based Extension

This extension is intentionally separate from the ligand-based benchmark in
`results/benchmark_results.csv`.

## Selected Target

- ChEMBL target: CHEMBL203
- Protein: Epidermal growth factor receptor (EGFR)
- Structure: RCSB PDB 1M17, EGFR tyrosine kinase domain with erlotinib
- Source: https://www.rcsb.org/structure/1M17

## Experiment Design

Use AutoDock Vina or an equivalent docking/scoring workflow to compare
structure-based scores with ligand-based ML predictions for the same compounds.

Suggested workflow:

1. Prepare the EGFR receptor from PDB 1M17.
2. Remove crystallographic water molecules and non-required heteroatoms.
3. Retain the co-crystallized ligand location to define the docking box.
4. Convert the receptor to PDBQT.
5. Select a subset of CHEMBL203 IC50 compounds from `data/processed/target_CHEMBL203_IC50.csv`.
6. Generate protonated 3D conformers for ligands.
7. Convert ligands to PDBQT.
8. Dock each ligand with the same box and Vina settings.
9. Export docking scores to `results/docking_egfr_1m17.csv`.
10. Join docking scores with `results/benchmark_results.csv` predictions for CHEMBL203.

## Reporting

Report docking as a separate structure-based experiment. Do not merge docking
scores into the ligand-based benchmark metrics. Recommended outputs:

- docking score summary
- Spearman correlation between docking score and pIC50
- Spearman correlation between docking score and ligand-based model prediction
- examples where docking and ligand-based ML disagree

## Notes

Docking scores are not direct binding free energies and should not be interpreted
as calibrated activity predictions without additional validation.
