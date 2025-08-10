import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.split import scaffold_split
from src.featurization.fingerprints import featurize_smiles_list
from src.models.cqr import CQR
from src.featurization.ood import ood_composite
from src.utils.io import save_joblib, save_json

def main():
    ap = argparse.ArgumentParser(description="Conformalized Quantile Regression training")
    ap.add_argument("--chembl_id", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--cal_pct", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (1-coverage)")
    args = ap.parse_args()

    csv_path = args.csv or f"data/processed/target_{args.chembl_id}.csv"
    df = pd.read_csv(csv_path).dropna(subset=["smiles","pIC50"]).reset_index(drop=True)

    # Scaffold-based split for realistic evaluation
    train_mask, test_mask = scaffold_split(df, "smiles", test_frac=0.2, seed=42)
    df_train, df_test = df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
    df_tr, df_cal = train_test_split(df_train, test_size=args.cal_pct, random_state=42)

    # Featurization
    X_tr, keep_tr = featurize_smiles_list(df_tr["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_tr = df_tr["pIC50"].values[keep_tr]
    smiles_tr = df_tr["smiles"].values[keep_tr]

    X_cal, keep_cal = featurize_smiles_list(df_cal["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_cal = df_cal["pIC50"].values[keep_cal]

    X_te, keep_te = featurize_smiles_list(df_test["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_te = df_test["pIC50"].values[keep_te]

    print(f"🎯 Training CQR on {args.chembl_id}")
    print(f"   Train: {len(X_tr)} | Cal: {len(X_cal)} | Test: {len(X_te)}")

    # Train CQR model
    cqr = CQR(alpha=args.alpha).fit(X_tr.astype(float), y_tr, X_cal.astype(float), y_cal)
    mu, lo, hi = cqr.predict_interval(X_te.astype(float))
    
    # Evaluate
    picp = float(((y_te >= lo) & (y_te <= hi)).mean())
    mpiw = float((hi - lo).mean())

    # OOD threshold (reuse composite from the other script)
    oods_tr = [ood_composite(x, X_tr) for x in X_tr]
    tau = float(np.quantile(oods_tr, 0.90))

    # Neighbor database for UI explanations
    neigh_db = {
        "smiles": smiles_tr.tolist(),
        "X": X_tr.astype(np.uint8),
        "y": y_tr.astype(float),
        "topk": 3
    }

    # Save artifacts
    outdir = Path(f"artifacts/{args.chembl_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    
    pack = {
        "type": "CQR_HGBR",
        "chembl_id": args.chembl_id,
        "bits": args.bits, "radius": args.radius,
        "alpha": args.alpha,
        "model": {"lq": cqr.lq_, "uq": cqr.uq_},
        "qhat_lo": cqr.qhat_lo_, 
        "qhat_hi": cqr.qhat_hi_,
        "ood_tau": tau,
        "X_train": X_tr, 
        "y_train": y_tr,
        "neighbors": neigh_db
    }
    
    save_joblib(outdir / "model_cqr.joblib", pack)
    save_json(outdir / "metrics_cqr.json", {
        "PICP": picp, 
        "MPIW": mpiw, 
        "quantile_corrections": {
            "qhat_lo": float(cqr.qhat_lo_),
            "qhat_hi": float(cqr.qhat_hi_)
        },
        "n_test": int(len(y_te))
    })
    
    print(f"[{args.chembl_id}] CQR-HGBR  PICP={picp:.3f} | MPIW={mpiw:.3f} | q̂_lo={cqr.qhat_lo_:.3f} | q̂_hi={cqr.qhat_hi_:.3f}")
    print(f"   Artifacts saved to: {outdir}")

if __name__ == "__main__":
    main()
