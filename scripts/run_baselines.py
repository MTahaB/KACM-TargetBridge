import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split import scaffold_split  # noqa: E402
from src.featurization.fingerprints import featurize_smiles_list  # noqa: E402
from src.models.knn_baseline import KNNRegressorTanimoto  # noqa: E402
from src.utils.io import save_json  # noqa: E402

ENDPOINT_COLUMNS = {"IC50": "pIC50", "KI": "pKi", "KD": "pKd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl_id", required=True)
    ap.add_argument("--endpoint", default="IC50", choices=sorted(ENDPOINT_COLUMNS))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    endpoint = args.endpoint.upper()
    y_col = ENDPOINT_COLUMNS[endpoint]
    csv_path = args.csv or f"data/processed/target_{args.chembl_id}_{endpoint}.csv"
    df = pd.read_csv(csv_path).dropna(subset=["smiles", y_col]).reset_index(drop=True)

    train_mask, test_mask = scaffold_split(df, "smiles", test_frac=0.2, seed=42)
    df_train, df_test = df[train_mask].reset_index(drop=True), df[
        test_mask
    ].reset_index(drop=True)

    X_tr, keep_tr = featurize_smiles_list(
        df_train["smiles"].tolist(), n_bits=args.bits, radius=args.radius
    )
    y_tr = df_train[y_col].values[keep_tr]
    X_te, keep_te = featurize_smiles_list(
        df_test["smiles"].tolist(), n_bits=args.bits, radius=args.radius
    )
    y_te = df_test[y_col].values[keep_te]

    knn = KNNRegressorTanimoto(k=args.k).fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    rho, _ = spearmanr(preds, y_te)
    mae = float(np.mean(np.abs(preds - y_te)))

    outdir = Path(f"artifacts/{args.chembl_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(
        outdir / f"baseline_knn_k{args.k}.json",
        {
            "endpoint": y_col,
            "spearman_rho": float(rho),
            "mae": mae,
            "n_test": int(len(y_te)),
        },
    )
    print(
        f"[{args.chembl_id}] kNN k={args.k} | "
        f"Spearman rho={float(rho):.3f} | MAE={mae:.3f} | n={len(y_te)}"
    )


if __name__ == "__main__":
    main()
