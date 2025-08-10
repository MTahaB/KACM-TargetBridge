#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini AlphaFold (Lite) – Binding Affinity (pIC50) Demo
-----------------------------------------------------
Ce script colle au challenge: sous-problème "Ligand–Protein Binding Affinity Estimation".
MVP: ingestion CSV -> featurisation légère -> petit modèle (KRR) par cible -> évaluation -> incertitude (conformal) -> démo prédiction.

Dépendances minimales:
    pip install pandas numpy scikit-learn matplotlib

Optionnel:
    pip install rdkit-pypi   # si dispo; sinon fallback auto sur hashing n-grams

Usage:
    python mini_affinity.py --csv mini_drug_target_dataset.csv --plot
    # ou juste:
    python mini_affinity.py   (le script créera un mini CSV si absent)
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

# =========================
# 0) Utils: Dataset demo
# =========================

DEMO_ROWS = [
    # ABL1
    {"smiles": "CC1=CC=CC=C1", "target": "ABL1", "pIC50": 8.0, "ligand": "Imatinib"},
    {"smiles": "CCOc1ccc2nc(SCc3ccccc3)sc2c1", "target": "ABL1", "pIC50": 7.5, "ligand": "Nilotinib"},
    {"smiles": "CNC(=O)c1csc(n1)Nc2ccc(cc2)F", "target": "ABL1", "pIC50": 6.5, "ligand": "Dasatinib"},
    # EGFR
    {"smiles": "COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OC", "target": "EGFR", "pIC50": 8.2, "ligand": "Gefitinib"},
    {"smiles": "COc1ccc2nc(N3CCNCC3)sc2c1", "target": "EGFR", "pIC50": 7.1, "ligand": "Erlotinib"},
    {"smiles": "COc1cccc2c1nc(N3CCN(CC3)C)nc2N", "target": "EGFR", "pIC50": 6.8, "ligand": "Afatinib"},
    # COX-2
    {"smiles": "CC1=CC=CC=C1C(=O)O", "target": "COX-2", "pIC50": 5.5, "ligand": "Aspirin"},
    {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "target": "COX-2", "pIC50": 6.2, "ligand": "Ibuprofen"},
    {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)Nc2ccc(cc2)S(=O)(=O)N", "target": "COX-2", "pIC50": 7.0, "ligand": "Celecoxib"},
]

def ensure_csv(csv_path: str) -> pd.DataFrame:
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        required = {"smiles", "target", "pIC50"}
        missing = required - set(df.columns.str.lower())
        # be forgiving with column case
        if "smiles" not in df.columns: df.rename(columns={c: "smiles" for c in df.columns if c.lower()=="smiles"}, inplace=True)
        if "target" not in df.columns: df.rename(columns={c: "target" for c in df.columns if c.lower()=="target"}, inplace=True)
        if "pIC50" not in df.columns and "pic50" in {c.lower() for c in df.columns}:
            df.rename(columns={c: "pIC50" for c in df.columns if c.lower()=="pic50"}, inplace=True)
        return df
    # create demo
    df = pd.DataFrame(DEMO_ROWS)
    df.to_csv(csv_path, index=False)
    print(f"[info] CSV demo créé: {csv_path}")
    return df

# =========================
# 1) Featurisation
# =========================

FEATURE_SIZE = 2048
USE_RDKIT = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    USE_RDKIT = False

def smiles_to_ecfp4(smiles: str, n_bits: int = FEATURE_SIZE) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros(n_bits, dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    for b in fp.GetOnBits():
        arr[b] = 1.0
    return arr

def smiles_to_hash_ngrams(smiles: str, n_bits: int = FEATURE_SIZE, n_min=2, n_max=3) -> np.ndarray:
    s = f"^{smiles}$"
    vec = np.zeros(n_bits, dtype=np.float32)
    for n in range(n_min, n_max+1):
        for i in range(len(s)-n+1):
            idx = hash(s[i:i+n]) % n_bits
            vec[idx] += 1.0
    # normalisation L2
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def featurize_series(smiles_series: pd.Series) -> np.ndarray:
    if USE_RDKIT:
        X = np.vstack([smiles_to_ecfp4(s) for s in smiles_series])
    else:
        X = np.vstack([smiles_to_hash_ngrams(s) for s in smiles_series])
    return X.astype(np.float32)

# Similarité pour OOD
def tanimoto_max_to_train(x: np.ndarray, X_train: np.ndarray) -> float:
    # Jaccard/Tanimoto pour vecteurs binaires {0,1}
    x = x.astype(bool)
    Xb = X_train.astype(bool)
    inter = (Xb & x).sum(axis=1).astype(float)
    union = (Xb | x).sum(axis=1).astype(float)
    sim = np.where(union>0, inter/union, 0.0)
    return float(sim.max()) if sim.size else 0.0

def cosine_max_to_train(x: np.ndarray, X_train: np.ndarray) -> float:
    denom = (np.linalg.norm(x) * np.linalg.norm(X_train, axis=1) + 1e-8)
    sims = X_train @ x / denom
    return float(sims.max()) if sims.size else 0.0

# =========================
# 2) Modélisation + calibration
# =========================

@dataclass
class TargetModel:
    model: KernelRidge
    X_train: np.ndarray  # pour OOD
    y_train: np.ndarray
    q90: float           # demi-largeur CP 90%

def fit_krr_with_conformal(X: np.ndarray, y: np.ndarray, random_state=42) -> TargetModel:
    # split train/calibration
    X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.3, random_state=random_state)
    # modèle simple (tu peux changer kernel/alpha)
    mdl = KernelRidge(alpha=1.0, kernel="linear")
    mdl.fit(X_tr, y_tr)
    y_cal_hat = mdl.predict(X_cal)
    # conformal split: quantile des erreurs absolues à 90%
    q90 = np.quantile(np.abs(y_cal - y_cal_hat), 0.9)
    return TargetModel(mdl, X_tr, y_tr, float(q90))

def evaluate_model(model: KernelRidge, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

def train_per_target(df: pd.DataFrame, random_state=42) -> Tuple[Dict[str, TargetModel], pd.DataFrame]:
    results = []
    models: Dict[str, TargetModel] = {}
    for tgt, dfg in df.groupby("target"):
        X = featurize_series(dfg["smiles"])
        y = dfg["pIC50"].astype(float).values
        # Hold-out pour reporting (indépendant de la calibration interne)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=random_state)
        mdl = KernelRidge(alpha=1.0, kernel="linear").fit(X_tr, y_tr)
        metrics = evaluate_model(mdl, X_te, y_te)
        # On refait un fit + calibration pour avoir q90 stocké
        TM = fit_krr_with_conformal(X, y, random_state)
        models[tgt] = TM
        results.append({
            "target": tgt, "n": len(dfg),
            "rmse": round(metrics["rmse"], 4),
            "mae": round(metrics["mae"], 4),
            "r2": round(metrics["r2"], 4),
            "use_rdkit": USE_RDKIT
        })
    return models, pd.DataFrame(results)

# =========================
# 3) Prédictions + OOD
# =========================

def predict_for_smiles(smiles: str, models: Dict[str, TargetModel], widen_on_ood=True) -> pd.DataFrame:
    x = featurize_series(pd.Series([smiles]))[0]
    rows = []
    for tgt, TM in models.items():
        y_hat = float(TM.model.predict(x.reshape(1,-1))[0])
        # OOD score = 1 - max similarity to training
        if USE_RDKIT:
            sim = tanimoto_max_to_train(x, TM.X_train)
        else:
            sim = cosine_max_to_train(x, TM.X_train)
        ood = 1.0 - sim
        half = TM.q90
        if widen_on_ood and ood > 0.5:  # seuil simple
            half *= 1.3
        lo, hi = y_hat - half, y_hat + half
        rows.append({"target": tgt, "pIC50_pred": y_hat, "ci90_lo": lo, "ci90_hi": hi, "ood": ood})
    out = pd.DataFrame(rows).sort_values("pIC50_pred", ascending=False).reset_index(drop=True)
    return out

# =========================
# 4) Plot util
# =========================

def plot_scatter_for_target(df: pd.DataFrame, tgt: str):
    dfg = df[df["target"] == tgt].copy()
    if dfg.empty:
        print(f"[warn] aucune donnée pour {tgt}")
        return
    X = featurize_series(dfg["smiles"])
    y = dfg["pIC50"].astype(float).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    mdl = KernelRidge(alpha=1.0, kernel="linear").fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)
    plt.figure()
    plt.scatter(y_te, y_pred)
    plt.xlabel("pIC50 (vrai)")
    plt.ylabel("pIC50 (prévu)")
    plt.title(f"{tgt} – prédiction vs vérité")
    plt.grid(True)
    plt.show()

# =========================
# 5) Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="mini_drug_target_dataset.csv",
                    help="Chemin du CSV (smiles,target,pIC50[,ligand])")
    ap.add_argument("--demo_smiles", type=str, default="CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
                    help="SMILES à tester (classement multi-cibles)")
    ap.add_argument("--plot", action="store_true", help="Afficher un scatter pour chaque cible")
    args = ap.parse_args()

    df = ensure_csv(args.csv)
    # Validation colonnes
    assert {"smiles", "target", "pIC50"}.issubset(df.columns), "CSV doit contenir smiles,target,pIC50"

    print(f"[info] USE_RDKIT={USE_RDKIT} | lignes={len(df)} | cibles={df['target'].nunique()}")
    models, report = train_per_target(df)
    print("\n=== Rapport par cible (hold-out) ===")
    print(report.to_string(index=False))

    print("\n=== Démo: classement multi-cibles pour ton SMILES ===")
    ranked = predict_for_smiles(args.demo_smiles, models, widen_on_ood=True)
    print(ranked.to_string(index=False))

    if args.plot:
        for tgt in sorted(df["target"].unique()):
            plot_scatter_for_target(df, tgt)

    print("\n[tip] Tu peux changer --demo_smiles pour tester d'autres molécules.")
    print("[tip] Pour le hackathon: ajoute un petit Streamlit autour de predict_for_smiles().")

if __name__ == "__main__":
    main()
