import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split import scaffold_split  # noqa: E402
from src.featurization.fingerprints import featurize_smiles_list  # noqa: E402
from src.models.conformal import AdaptiveConformalRegressor  # noqa: E402
from src.models.cqr import CQR  # noqa: E402
from src.models.knn_baseline import KNNRegressorTanimoto  # noqa: E402
from src.models.krr import TanimotoKRR  # noqa: E402
from src.utils.chem import murcko_scaffold_smiles  # noqa: E402

ENDPOINT_COLUMNS = {"IC50": "pIC50", "KI": "pKi", "KD": "pKd"}
TARGET_METADATA = {
    "CHEMBL203": {
        "target_name": "Epidermal growth factor receptor",
        "organism": "Homo sapiens",
    },
    "CHEMBL1862": {
        "target_name": "Tyrosine-protein kinase ABL1",
        "organism": "Homo sapiens",
    },
    "CHEMBL217": {
        "target_name": "D(2) dopamine receptor",
        "organism": "Homo sapiens",
    },
}
DEFAULT_MODELS = ["knn", "random_forest", "krr_conformal", "cqr"]


def endpoint_column(endpoint: str) -> str:
    endpoint = endpoint.upper()
    if endpoint not in ENDPOINT_COLUMNS:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    return ENDPOINT_COLUMNS[endpoint]


def load_target_data(data_dir: Path, target_id: str, endpoint: str) -> pd.DataFrame:
    y_col = endpoint_column(endpoint)
    path = data_dir / f"target_{target_id}_{endpoint.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/download_chembl.py for this target "
            "and endpoint."
        )
    df = pd.read_csv(path).dropna(subset=["smiles", y_col]).reset_index(drop=True)
    return df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)


def split_target(df: pd.DataFrame, seed: int, test_frac: float, cal_frac: float):
    train_cal_mask, test_mask = scaffold_split(
        df, "smiles", test_frac=test_frac, seed=seed
    )
    df_train_cal = df[train_cal_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)
    train_mask, cal_mask = scaffold_split(
        df_train_cal, "smiles", test_frac=cal_frac, seed=seed + 1
    )
    df_train = df_train_cal[train_mask].reset_index(drop=True)
    df_cal = df_train_cal[cal_mask].reset_index(drop=True)
    return df_train, df_cal, df_test


def scaffold_count(smiles):
    scaffolds = [
        murcko_scaffold_smiles(smi) or f"NOSCAF_{i}" for i, smi in enumerate(smiles)
    ]
    return len(set(scaffolds))


def featurize_split(df: pd.DataFrame, y_col: str, bits: int, radius: int):
    X, keep = featurize_smiles_list(df["smiles"].tolist(), n_bits=bits, radius=radius)
    y = df[y_col].to_numpy(dtype=float)[keep]
    smiles = df["smiles"].to_numpy()[keep]
    return X, y, smiles


def metric_row(y_true, y_pred, lo=None, hi=None):
    rho, _ = spearmanr(y_pred, y_true)
    row = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "picp": np.nan,
        "mpiw": np.nan,
    }
    if lo is not None and hi is not None:
        row["picp"] = float(((y_true >= lo) & (y_true <= hi)).mean())
        row["mpiw"] = float(np.mean(hi - lo))
    return row


def base_row(args, target_id, endpoint, y_col, df, splits, split_scaffolds, model):
    metadata = TARGET_METADATA.get(target_id, {})
    return {
        "target_chembl_id": target_id,
        "target_name": metadata.get("target_name", ""),
        "organism": metadata.get("organism", ""),
        "endpoint": y_col,
        "activity_type": endpoint.upper(),
        "endpoint_definition": "-log10(activity in molar units)",
        "filter_relation": "=",
        "n_compounds": int(len(df)),
        "n_train": int(splits["train"]),
        "n_calibration": int(splits["calibration"]),
        "n_test": int(splits["test"]),
        "n_train_scaffolds": int(split_scaffolds["train"]),
        "n_calibration_scaffolds": int(split_scaffolds["calibration"]),
        "n_test_scaffolds": int(split_scaffolds["test"]),
        "split_type": "Murcko scaffold split",
        "seed": int(args.seed),
        "test_frac": float(args.test_frac),
        "cal_frac": float(args.cal_frac),
        "fingerprint": "Morgan binary fingerprint",
        "bits": int(args.bits),
        "radius": int(args.radius),
        "model": model,
    }


def run_models(args, target_id: str):
    endpoint = args.endpoint.upper()
    y_col = endpoint_column(endpoint)
    df = load_target_data(Path(args.data_dir), target_id, endpoint)
    df_train, df_cal, df_test = split_target(
        df, seed=args.seed, test_frac=args.test_frac, cal_frac=args.cal_frac
    )

    X_train, y_train, _ = featurize_split(df_train, y_col, args.bits, args.radius)
    X_cal, y_cal, _ = featurize_split(df_cal, y_col, args.bits, args.radius)
    X_test, y_test, _ = featurize_split(df_test, y_col, args.bits, args.radius)

    splits = {
        "train": len(y_train),
        "calibration": len(y_cal),
        "test": len(y_test),
    }
    split_scaffolds = {
        "train": scaffold_count(df_train["smiles"]),
        "calibration": scaffold_count(df_cal["smiles"]),
        "test": scaffold_count(df_test["smiles"]),
    }

    rows = []
    models = set(args.models)

    if "knn" in models:
        start = time.perf_counter()
        model = KNNRegressorTanimoto(k=args.knn_k).fit(X_train, y_train)
        pred = model.predict(X_test)
        row = base_row(
            args,
            target_id,
            endpoint,
            y_col,
            df,
            splits,
            split_scaffolds,
            "k-NN",
        )
        row.update(metric_row(y_test, pred))
        row["runtime_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)

    if "random_forest" in models:
        start = time.perf_counter()
        model = RandomForestRegressor(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        ).fit(X_train.astype(float), y_train)
        pred = model.predict(X_test.astype(float))
        row = base_row(
            args,
            target_id,
            endpoint,
            y_col,
            df,
            splits,
            split_scaffolds,
            "Random Forest",
        )
        row.update(metric_row(y_test, pred))
        row["runtime_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)

    if "krr_conformal" in models:
        start = time.perf_counter()
        base = TanimotoKRR(alpha=args.krr_alpha)
        model = AdaptiveConformalRegressor(
            model=base,
            alpha=args.miscoverage,
            gamma=args.conformal_gamma,
            k_dens=8,
        ).fit_calibrate(X_train, y_train, X_cal, y_cal)
        pred, lo, hi = model.predict_interval(X_test)
        row = base_row(
            args,
            target_id,
            endpoint,
            y_col,
            df,
            splits,
            split_scaffolds,
            "KRR-Conformal",
        )
        row.update(metric_row(y_test, pred, lo, hi))
        row["runtime_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)

    if "cqr" in models:
        start = time.perf_counter()
        model = CQR(alpha=args.miscoverage).fit(
            X_train.astype(float), y_train, X_cal.astype(float), y_cal
        )
        pred, lo, hi = model.predict_interval(X_test.astype(float))
        row = base_row(
            args,
            target_id,
            endpoint,
            y_col,
            df,
            splits,
            split_scaffolds,
            "CQR",
        )
        row.update(metric_row(y_test, pred, lo, hi))
        row["runtime_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)

    if "xgboost" in models:
        start = time.perf_counter()
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=args.xgb_trees,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
        ).fit(X_train.astype(float), y_train)
        pred = model.predict(X_test.astype(float))
        row = base_row(
            args,
            target_id,
            endpoint,
            y_col,
            df,
            splits,
            split_scaffolds,
            "XGBoost",
        )
        row.update(metric_row(y_test, pred))
        row["runtime_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmark models with shared target splits."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["CHEMBL203", "CHEMBL1862", "CHEMBL217"],
    )
    parser.add_argument("--endpoint", default="IC50", choices=sorted(ENDPOINT_COLUMNS))
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out", default="results/benchmark_results.csv")
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--cal_frac", type=float, default=0.2)
    parser.add_argument("--miscoverage", type=float, default=0.1)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--rf_trees", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--krr_alpha", type=float, default=1.0)
    parser.add_argument("--conformal_gamma", type=float, default=1.5)
    parser.add_argument("--xgb_trees", type=int, default=400)
    parser.add_argument("--xgb_max_depth", type=int, default=4)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.03)
    args = parser.parse_args()

    invalid = sorted(set(args.models) - set(DEFAULT_MODELS + ["xgboost"]))
    if invalid:
        raise ValueError(f"Unknown model names: {invalid}")

    rows = []
    for target_id in args.targets:
        print(f"Benchmarking {target_id} {args.endpoint.upper()}")
        rows.extend(run_models(args, target_id))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out, index=False)
    print(f"Wrote {out} - {len(results)} rows")


if __name__ == "__main__":
    main()
