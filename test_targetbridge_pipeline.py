# test_targetbridge_pipeline.py
import numpy as np
import pandas as pd
from core import (TARGETS, load_tables, morgan_fp, ecfp_matrix, fit_krr,
                  split_conformal_q, nearest_ligand_info, ood_score_against_target)
from sklearn.model_selection import train_test_split

def run_once(query_smiles: str, alpha: float = 0.10, ood_thresh: float = 0.30):
    tables = load_tables()
    if not tables:
        print("No tables. Run scripts/fetch_chembl.py first."); return

    qfp = morgan_fp(query_smiles)
    if qfp is None:
        print("Invalid SMILES:", query_smiles); return

    results = []
    for tid, df in tables.items():
        X = ecfp_matrix(df["smiles"].tolist())
        if X is None: continue
        y = df["pIC50"].to_numpy()
        if len(y) != X.shape[0]: y = y[:X.shape[0]]

        Xtr, Xcal, ytr, ycal = train_test_split(X, y, test_size=0.2, random_state=42)
        model = fit_krr(Xtr, ytr, alpha=1.0)
        q = split_conformal_q(model, Xcal, ycal, alpha=alpha)

        mu = float(model.predict(qfp.reshape(1,-1).astype(np.float32))[0])
        ood = ood_score_against_target(qfp, df)
        lo, hi = mu - q, mu + q
        badge = "OOD" if ood >= ood_thresh else ""
        w_smi, w_y, w_sim = nearest_ligand_info(qfp, df)
        why = f"{w_smi} (pIC50={w_y:.2f}) | sim={w_sim:.2f}" if w_smi else "—"
        score = mu + q

        results.append((tid, mu, lo, hi, badge, score, why))

    if not results:
        print("No results."); return

    results.sort(key=lambda r: r[5], reverse=True)
    out = pd.DataFrame(results, columns=["Target","pIC50_pred","CI_lo","CI_hi","OOD","score","why"])
    print("\n=== TargetBridge (offline test) ===")
    print("Query SMILES:", query_smiles)
    print(out.to_string(index=False, justify='left', max_colwidth=60))

if __name__ == "__main__":
    run_once("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")  # ibuprofen
