# KACM-TargetBridge: Uncertainty-Aware Molecular Activity Prediction from ChEMBL Data

KACM-TargetBridge is a ligand-based molecular activity prediction project using
ChEMBL bioactivity data, Morgan fingerprints, scaffold-based evaluation splits,
and empirical uncertainty evaluation.

The benchmark reported here uses IC50 measurements only. IC50, Ki, and Kd are not
merged. Separate endpoint experiments can be run with the downloader by selecting
`IC50`, `KI`, or `KD`, which produce `pIC50`, `pKi`, or `pKd` tables respectively.

All numeric values reported in this README are taken from
[`results/benchmark_results.csv`](results/benchmark_results.csv).

## Repository Layout

```text
scripts/download_chembl.py        Download one target and one endpoint from ChEMBL
scripts/benchmark.py              Run all benchmark models on shared splits
scripts/train_per_target.py       Train KRR-conformal artifacts for the Streamlit app
scripts/train_per_target_cqr.py   Train CQR artifacts for the Streamlit app
src/data/split.py                 Murcko scaffold split utilities
src/featurization/                Molecular fingerprints and OOD utilities
src/models/                       k-NN, KRR, conformal, CQR, and ensemble code
src/ui/app.py                     Streamlit prediction interface
results/benchmark_results.csv     Benchmark outputs used by this README
```

## Data Definition

The benchmark uses ChEMBL activities with:

- `standard_type = IC50`
- `standard_relation = =`
- units convertible to molar concentration
- duplicate SMILES aggregated by median endpoint value

The endpoint is `pIC50`, defined in the results file as
`-log10(activity in molar units)`.

## Targets And Splits

Splits use Murcko scaffolds with `seed = 42`, `test_frac = 0.2`, and
`cal_frac = 0.2`. Calibration compounds are scaffold-separated from the training
subset and the test set contains unseen scaffolds relative to the training and
calibration subsets.

| Target | Name | Endpoint | Compounds | Train | Calibration | Test |
|---|---|---:|---:|---:|---:|---:|
| CHEMBL203 | Epidermal growth factor receptor | pIC50 | 10834 | 6933 | 1733 | 2168 |
| CHEMBL1862 | Tyrosine-protein kinase ABL1 | pIC50 | 2151 | 1374 | 344 | 433 |
| CHEMBL217 | D(2) dopamine receptor | pIC50 | 890 | 567 | 142 | 181 |

The molecular representation is a Morgan binary fingerprint with `bits = 2048`
and `radius = 2`.

## Models

The benchmark compares:

- k-NN with Tanimoto similarity
- Random Forest on Morgan fingerprints
- KRR-Conformal, using Tanimoto kernel ridge regression with split conformal
  intervals
- CQR, using quantile regression with conformal calibration

Point-prediction metrics are MAE, RMSE, and Spearman correlation. Interval
metrics are PICP and MPIW and are reported for interval-producing methods.

## Benchmark Results

| Target | Model | MAE | RMSE | Spearman | PICP | MPIW |
|---|---|---:|---:|---:|---:|---:|
| CHEMBL203 | k-NN | 0.685 | 0.915 | 0.670 |  |  |
| CHEMBL203 | Random Forest | 0.664 | 0.870 | 0.714 |  |  |
| CHEMBL203 | KRR-Conformal | 0.689 | 0.904 | 0.707 | 0.901 | 2.831 |
| CHEMBL203 | CQR | 0.868 | 1.077 | 0.538 | 0.923 | 3.667 |
| CHEMBL1862 | k-NN | 0.541 | 0.743 | 0.855 |  |  |
| CHEMBL1862 | Random Forest | 0.562 | 0.757 | 0.842 |  |  |
| CHEMBL1862 | KRR-Conformal | 0.577 | 0.750 | 0.841 | 0.931 | 2.623 |
| CHEMBL1862 | CQR | 0.669 | 0.889 | 0.805 | 0.845 | 2.425 |
| CHEMBL217 | k-NN | 0.737 | 1.012 | 0.674 |  |  |
| CHEMBL217 | Random Forest | 0.721 | 1.155 | 0.672 |  |  |
| CHEMBL217 | KRR-Conformal | 0.780 | 1.212 | 0.630 | 0.978 | 4.347 |
| CHEMBL217 | CQR | 0.932 | 1.191 | 0.464 | 0.983 | 4.570 |

Interval coverage is empirically evaluated on unseen scaffolds. It should not be
interpreted as a formal distribution-free coverage guarantee under scaffold
shift.

## Reproducing The Benchmark

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Download IC50-only data:

```bash
python scripts/download_chembl.py --chembl_id CHEMBL203 --activity_type IC50
python scripts/download_chembl.py --chembl_id CHEMBL1862 --activity_type IC50
python scripts/download_chembl.py --chembl_id CHEMBL217 --activity_type IC50
```

Run the shared-split benchmark:

```bash
python scripts/benchmark.py --targets CHEMBL203 CHEMBL1862 CHEMBL217 --endpoint IC50 --out results/benchmark_results.csv
```

## Streamlit Interface

The Streamlit app loads trained artifacts from `artifacts/`. Those files are
generated outputs and are not tracked in Git. To regenerate app artifacts for the
IC50 endpoint, run:

```bash
python scripts/train_per_target.py --chembl_id CHEMBL203 --endpoint IC50
python scripts/train_per_target.py --chembl_id CHEMBL1862 --endpoint IC50
python scripts/train_per_target.py --chembl_id CHEMBL217 --endpoint IC50
python scripts/train_per_target_cqr.py --chembl_id CHEMBL203 --endpoint IC50
python scripts/train_per_target_cqr.py --chembl_id CHEMBL1862 --endpoint IC50
python scripts/train_per_target_cqr.py --chembl_id CHEMBL217 --endpoint IC50
```

Then launch:

```bash
streamlit run src/ui/app.py
```

## Limitations

This project is ligand-based. It does not currently use protein structures,
docking poses, binding-site information, molecular dynamics, or target sequence
models.

The benchmark evaluates scaffold generalization within curated ChEMBL endpoint
tables. It does not establish prospective performance on newly synthesized
compounds.

Future work includes adding protein structures, docking or scoring experiments,
and generative molecular design workflows. A structure-based extension should be
reported separately from the ligand-based benchmark so that docking scores and
fingerprint-based predictions are not conflated.

An optional structure-based EGFR docking protocol is outlined in
[`docs/structure_based_extension.md`](docs/structure_based_extension.md).
