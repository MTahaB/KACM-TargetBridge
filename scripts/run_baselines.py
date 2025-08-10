import argparse
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.split import scaffold_split
from src.featurization.fingerprints import featurize_smiles_list
from src.models.knn_baseline import KNNRegressorTanimoto
from src.utils.io import save_json
from scipy.stats import spearmanr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl_id", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    csv_path = args.csv or f"data/processed/target_{args.chembl_id}.csv"
    df = pd.read_csv(csv_path).dropna(subset=["smiles", "pIC50"]).reset_index(drop=True)

    train_mask, test_mask = scaffold_split(df, "smiles", test_frac=0.2, seed=42)
    df_train, df_test = df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)

    X_tr, keep_tr = featurize_smiles_list(df_train["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_tr = df_train["pIC50"].values[keep_tr]
    X_te, keep_te = featurize_smiles_list(df_test["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_te = df_test["pIC50"].values[keep_te]

    knn = KNNRegressorTanimoto(k=args.k).fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    rho, _ = spearmanr(preds, y_te)
    mae = float(np.mean(np.abs(preds - y_te)))

    outdir = Path(f"artifacts/{args.chembl_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / f"baseline_knn_k{args.k}.json", {"spearman_rho": float(rho), "mae": mae, "n_test": int(len(y_te))})
    print(f"[{args.chembl_id}] kNN k={args.k} | Spearman ρ={float(rho):.3f} | MAE={mae:.3f} | n={len(y_te)}")

if __name__ == "__main__":
    main()
