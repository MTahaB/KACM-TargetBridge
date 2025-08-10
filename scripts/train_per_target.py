import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.split import scaffold_split
from src.featurization.fingerprints import featurize_smiles_list, morgan_fp
from src.models.krr import TanimotoKRR
from src.models.conformal import AdaptiveConformalRegressor
from src.featurization.ood import ood_composite, tanimoto_sim_matrix
from src.utils.io import save_joblib, save_json

def build_neighbors_db(smiles_list, X, y, topk=3):
    """Prépare un mini index pour l'explication kNN (renvoyé en UI)."""
    return {
        "smiles": smiles_list,
        "X": X.astype(np.uint8),
        "y": y.astype(float),
        "topk": int(topk)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl_id", required=True)
    ap.add_argument("--csv", default=None, help="Optional CSV: smiles,pIC50")
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cal_pct", type=float, default=0.2)
    ap.add_argument("--ood_tau", type=float, default=0.55, help="Seuil OOD composite (↑ plus OOD)")
    ap.add_argument("--gamma", type=float, default=1.5, help="Amplification adaptative des intervalles")
    ap.add_argument("--z_gp", type=float, default=1.0, help="poids σ_GP dans l'intervalle hybride")
    args = ap.parse_args()

    csv_path = args.csv or f"data/processed/target_{args.chembl_id}.csv"
    df = pd.read_csv(csv_path).dropna(subset=["smiles","pIC50"]).reset_index(drop=True)

    # Split scaffold pour crédibilité
    train_mask, test_mask = scaffold_split(df, "smiles", test_frac=0.2, seed=42)
    df_train, df_test = df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
    df_tr, df_cal = train_test_split(df_train, test_size=args.cal_pct, random_state=42)

    # Featurisation
    X_tr, keep_tr = featurize_smiles_list(df_tr["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_tr = df_tr["pIC50"].values[keep_tr]
    smiles_tr = df_tr["smiles"].values[keep_tr]

    X_cal, keep_cal = featurize_smiles_list(df_cal["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_cal = df_cal["pIC50"].values[keep_cal]

    X_te, keep_te = featurize_smiles_list(df_test["smiles"].tolist(), n_bits=args.bits, radius=args.radius)
    y_te = df_test["pIC50"].values[keep_te]

    # Entraînement + Conformal adaptatif
    base = TanimotoKRR(alpha=args.alpha)
    ac = AdaptiveConformalRegressor(model=base, alpha=0.1, gamma=args.gamma, k_dens=8).fit_calibrate(X_tr, y_tr, X_cal, y_cal)

    # Conformal adaptatif
    mu_c, lo_c, hi_c = ac.predict_interval(X_te)
    picp = float(((y_te >= lo_c) & (y_te <= hi_c)).mean())
    mpiw = float((hi_c - lo_c).mean())

    # Variance GP du KRR
    mu_gp, var_gp = ac.model.predict_mean_var(X_te)
    sig_gp = np.sqrt(var_gp)

    # Intervalle hybride (affiché dans l'UI si on veut)
    lo_h = mu_c - args.z_gp * sig_gp
    hi_h = mu_c + args.z_gp * sig_gp

    # Seuil OOD via quantile entraînement (conservateur)
    # On calcule l’OOD composite sur train et fixe tau au 90e percentile si pas fourni
    oods_tr = [ood_composite(x, X_tr) for x in X_tr]
    tau = args.ood_tau if args.ood_tau is not None else float(np.quantile(oods_tr, 0.90))

    # Voisins pour explication (UI calculera à la volée)
    neigh_db = build_neighbors_db(smiles_tr.tolist(), X_tr, y_tr, topk=3)

    outdir = Path(f"artifacts/{args.chembl_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    pack = {
        "type": "KRR_Tanimoto_AdaptiveConformal+GPVar",
        "chembl_id": args.chembl_id,
        "bits": args.bits, "radius": args.radius,
        "alpha": args.alpha,
        "alpha_cp": 0.1,
        "gamma": args.gamma,
        "ood_tau": tau,
        "model": ac.model,            # contient L_ pour variance
        "qhat": ac.qhat_,
        "X_train": X_tr,
        "y_train": y_tr,
        "neighbors": neigh_db,
        "z_gp": args.z_gp
    }
    save_joblib(outdir / "model.joblib", pack)
    save_json(outdir / "metrics.json", {
        "PICP_conformal": picp, 
        "MPIW_conformal": mpiw,
        "mean_sigma_gp": float(sig_gp.mean()), 
        "n_test": int(len(y_te))
    })
    print(f"[{args.chembl_id}] PICP={picp:.3f} | MPIW={mpiw:.3f} | σ_GP~{sig_gp.mean():.3f} | OOD_tau≈{tau:.2f} | artifacts -> {outdir}")

if __name__ == "__main__":
    main()
