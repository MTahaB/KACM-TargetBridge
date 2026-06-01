"""Train per-target prediction artifacts.

The script supports KRR-conformal, CQR, and ensemble artifacts through a single
``--method`` flag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split import scaffold_split  # noqa: E402
from src.featurization.fingerprints import featurize_smiles_list  # noqa: E402
from src.featurization.ood import ood_composite  # noqa: E402
from src.models.conformal import AdaptiveConformalRegressor  # noqa: E402
from src.models.cqr import CQR  # noqa: E402
from src.models.ensemble import StackedEnsemble  # noqa: E402
from src.models.krr import TanimotoKRR  # noqa: E402
from src.utils.io import save_joblib, save_json  # noqa: E402

ENDPOINT_COLUMNS = {"IC50": "pIC50", "KI": "pKi", "KD": "pKd"}
FEATURE_SETS = ["morgan", "morgan_rdkit"]


def endpoint_column(endpoint: str) -> str:
    """Map an assay endpoint name to the corresponding p-value column."""
    endpoint = endpoint.upper()
    if endpoint not in ENDPOINT_COLUMNS:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    return ENDPOINT_COLUMNS[endpoint]


def build_neighbors_db(smiles_list, X: np.ndarray, y: np.ndarray, topk: int = 3):
    """Build the compact nearest-neighbor payload used by the Streamlit UI."""
    return {
        "smiles": list(smiles_list),
        "X": X.astype(np.uint8),
        "y": y.astype(float),
        "topk": int(topk),
    }


def load_splits(args: argparse.Namespace):
    """Load and split one target table.

    Returns
    -------
    tuple
        DataFrames for train, calibration, and test plus the endpoint column.
    """
    endpoint = args.endpoint.upper()
    y_col = endpoint_column(endpoint)
    csv_path = args.csv or f"data/processed/target_{args.chembl_id}_{endpoint}.csv"
    df = pd.read_csv(csv_path).dropna(subset=["smiles", y_col]).reset_index(drop=True)
    train_mask, test_mask = scaffold_split(df, "smiles", test_frac=0.2, seed=args.seed)
    df_train, df_test = df[train_mask].reset_index(drop=True), df[
        test_mask
    ].reset_index(drop=True)
    df_tr, df_cal = train_test_split(
        df_train, test_size=args.cal_pct, random_state=args.seed
    )
    return df_tr, df_cal, df_test, y_col


def featurize(
    df: pd.DataFrame,
    y_col: str,
    args: argparse.Namespace,
    feature_set: str = "morgan",
):
    """Featurize one split and align labels to valid molecules."""
    X, keep = featurize_smiles_list(
        df["smiles"].tolist(),
        n_bits=args.bits,
        radius=args.radius,
        feature_set=feature_set,
    )
    y = df[y_col].to_numpy(dtype=float)[keep]
    smiles = df["smiles"].to_numpy()[keep]
    return X, y, smiles


def ood_threshold(X_train: np.ndarray, override: float | None) -> float:
    """Return the OOD threshold used in UI artifact metadata."""
    if override is not None:
        return float(override)
    return float(np.quantile([ood_composite(x, X_train) for x in X_train], 0.90))


def save_metrics(outdir: Path, name: str, metrics: dict[str, Any]) -> None:
    """Persist a metrics JSON file."""
    save_json(outdir / name, metrics)


def train_krr(args: argparse.Namespace) -> None:
    """Train a Tanimoto KRR model with adaptive conformal intervals."""
    df_tr, df_cal, df_test, y_col = load_splits(args)
    X_tr, y_tr, smiles_tr = featurize(df_tr, y_col, args)
    X_cal, y_cal, _ = featurize(df_cal, y_col, args)
    X_te, y_te, _ = featurize(df_test, y_col, args)

    base = TanimotoKRR(alpha=args.alpha)
    ac = AdaptiveConformalRegressor(
        model=base, alpha=args.miscoverage, gamma=args.gamma, k_dens=8
    ).fit_calibrate(X_tr, y_tr, X_cal, y_cal)
    mu, lo, hi = ac.predict_interval(X_te)
    _, var_gp = ac.model.predict_mean_var(X_te)
    sig_gp = np.sqrt(var_gp)
    tau = ood_threshold(X_tr, args.ood_tau)

    outdir = Path(args.outdir) / args.chembl_id
    outdir.mkdir(parents=True, exist_ok=True)
    save_joblib(
        outdir / "model.joblib",
        {
            "type": "KRR_Tanimoto_AdaptiveConformal+GPVar",
            "chembl_id": args.chembl_id,
            "endpoint": y_col,
            "bits": args.bits,
            "radius": args.radius,
            "alpha": args.alpha,
            "alpha_cp": args.miscoverage,
            "gamma": args.gamma,
            "feature_set": "morgan",
            "ood_tau": tau,
            "model": ac.model,
            "qhat": ac.qhat_,
            "X_train": X_tr,
            "y_train": y_tr,
            "neighbors": build_neighbors_db(smiles_tr.tolist(), X_tr, y_tr),
        },
    )
    save_metrics(
        outdir,
        "metrics.json",
        {
            "PICP_conformal": float(((y_te >= lo) & (y_te <= hi)).mean()),
            "MPIW_conformal": float((hi - lo).mean()),
            "mean_sigma_gp": float(sig_gp.mean()),
            "n_test": int(len(y_te)),
        },
    )
    print(f"[{args.chembl_id}] krr artifact -> {outdir}")


def train_cqr(args: argparse.Namespace) -> None:
    """Train a conformalized quantile regression artifact."""
    df_tr, df_cal, df_test, y_col = load_splits(args)
    X_tr, y_tr, smiles_tr = featurize(df_tr, y_col, args, args.feature_set)
    X_cal, y_cal, _ = featurize(df_cal, y_col, args, args.feature_set)
    X_te, y_te, _ = featurize(df_test, y_col, args, args.feature_set)
    X_tr_ood, _, _ = featurize(df_tr, y_col, args, "morgan")

    cqr = CQR(alpha=args.miscoverage).fit(
        X_tr.astype(float), y_tr, X_cal.astype(float), y_cal
    )
    _, lo, hi = cqr.predict_interval(X_te.astype(float))
    tau = ood_threshold(X_tr_ood, args.ood_tau)

    outdir = Path(args.outdir) / args.chembl_id
    outdir.mkdir(parents=True, exist_ok=True)
    save_joblib(
        outdir / "model_cqr.joblib",
        {
            "type": "CQR_HGBR",
            "chembl_id": args.chembl_id,
            "endpoint": y_col,
            "bits": args.bits,
            "radius": args.radius,
            "alpha": args.miscoverage,
            "feature_set": args.feature_set,
            "model": {"lq": cqr.lq_, "uq": cqr.uq_},
            "qhat": cqr.qhat_,
            "qhat_lo": cqr.qhat_lo_,
            "qhat_hi": cqr.qhat_hi_,
            "ood_tau": tau,
            "X_train": X_tr_ood,
            "y_train": y_tr,
            "neighbors": build_neighbors_db(smiles_tr.tolist(), X_tr_ood, y_tr),
        },
    )
    save_metrics(
        outdir,
        "metrics_cqr.json",
        {
            "PICP": float(((y_te >= lo) & (y_te <= hi)).mean()),
            "MPIW": float((hi - lo).mean()),
            "qhat": float(cqr.qhat_),
            "n_test": int(len(y_te)),
        },
    )
    print(f"[{args.chembl_id}] cqr artifact -> {outdir}")


def train_ensemble(args: argparse.Namespace) -> None:
    """Train a stacked ensemble artifact with uncertainty from disagreement."""
    df_tr, df_cal, df_test, y_col = load_splits(args)
    X_tr, y_tr, smiles_tr = featurize(df_tr, y_col, args)
    X_cal, y_cal, _ = featurize(df_cal, y_col, args)
    X_te, y_te, _ = featurize(df_test, y_col, args)

    model = StackedEnsemble(
        krr_alpha=args.alpha,
        rf_n_estimators=args.ensemble_trees,
        ridge_alpha=args.ridge_alpha,
    ).fit(X_tr, y_tr, X_cal, y_cal)
    pred, unc = model.predict_with_uncertainty(X_te)
    lo, hi = pred - args.ensemble_z * unc, pred + args.ensemble_z * unc
    tau = ood_threshold(X_tr, args.ood_tau)

    outdir = Path(args.outdir) / args.chembl_id
    outdir.mkdir(parents=True, exist_ok=True)
    save_joblib(
        outdir / "model_ensemble.joblib",
        {
            "type": "StackedEnsemble",
            "chembl_id": args.chembl_id,
            "endpoint": y_col,
            "bits": args.bits,
            "radius": args.radius,
            "feature_set": "morgan",
            "model": model,
            "ensemble_z": args.ensemble_z,
            "ood_tau": tau,
            "X_train": X_tr,
            "y_train": y_tr,
            "neighbors": build_neighbors_db(smiles_tr.tolist(), X_tr, y_tr),
        },
    )
    save_metrics(
        outdir,
        "metrics_ensemble.json",
        {
            "PICP": float(((y_te >= lo) & (y_te <= hi)).mean()),
            "MPIW": float((hi - lo).mean()),
            "MAE": float(np.mean(np.abs(pred - y_te))),
            "n_test": int(len(y_te)),
        },
    )
    print(f"[{args.chembl_id}] ensemble artifact -> {outdir}")


def main() -> None:
    """Parse arguments and train the selected method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl_id", required=True)
    parser.add_argument("--endpoint", default="IC50", choices=sorted(ENDPOINT_COLUMNS))
    parser.add_argument("--method", default="krr", choices=["krr", "cqr", "ensemble"])
    parser.add_argument(
        "--feature_set",
        default="morgan",
        choices=FEATURE_SETS,
        help="Feature set for CQR artifacts. KRR and ensemble use Morgan features.",
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--miscoverage", type=float, default=0.1)
    parser.add_argument("--cal_pct", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--ood_tau", type=float, default=None)
    parser.add_argument("--ensemble_trees", type=int, default=100)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--ensemble_z", type=float, default=1.64)
    args = parser.parse_args()

    if args.method == "krr":
        train_krr(args)
    elif args.method == "cqr":
        train_cqr(args)
    else:
        train_ensemble(args)


if __name__ == "__main__":
    main()
