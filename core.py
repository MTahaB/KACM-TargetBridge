from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rdfp

from sklearn.kernel_ridge import KernelRidge

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path("data")
TARGETS = ["ABL1", "EGFR", "PTGS2", "DRD2"]

# ECFP4: radius=2, 2048 bits
_GEN = rdfp.GetMorganGenerator(radius=2, fpSize=2048)

def morgan_fp(smiles: str, nbits: int = 2048):
    """Return a single numpy array (2048 bits) or None if SMILES invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _GEN.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def tanimoto_bits(a_bits: np.ndarray, b_bits: np.ndarray) -> float:
    bv_a = DataStructs.ExplicitBitVect(len(a_bits))
    bv_a.SetBitsFromList([i for i, v in enumerate(a_bits) if v])
    bv_b = DataStructs.ExplicitBitVect(len(b_bits))
    bv_b.SetBitsFromList([i for i, v in enumerate(b_bits) if v])
    return float(DataStructs.TanimotoSimilarity(bv_a, bv_b))

def ecfp_matrix(smiles_list, nbits=2048):
    """Return a (n,2048) float32 matrix, or None if nothing valid."""
    fps = []
    for s in smiles_list:
        fp = morgan_fp(str(s), nbits=nbits)
        if fp is not None:
            fps.append(fp)
    return np.vstack(fps).astype(np.float32) if fps else None

def fit_krr(X_train, y_train, alpha=1.0):
    """Kernel Ridge (RBF) with gamma=1/d."""
    mdl = KernelRidge(alpha=alpha, kernel="rbf", gamma=1.0 / X_train.shape[1])
    mdl.fit(X_train, y_train)
    return mdl

def split_conformal_q(model, X_cal, y_cal, alpha=0.10):
    r = np.abs(y_cal - model.predict(X_cal))
    return float(np.quantile(r, 1 - alpha))

def nearest_ligand_info(query_fp, df_target):
    best = ("", None, -1.0)
    for _, r in df_target.iterrows():
        fp = morgan_fp(r["smiles"])
        if fp is None:
            continue
        sim = tanimoto_bits(query_fp, fp)
        if sim > best[2]:
            best = (r["smiles"], float(r.get("pIC50", np.nan)), sim)
    return best

def ood_score_against_target(query_fp, df_target):
    sims = []
    for s in df_target["smiles"]:
        fp = morgan_fp(s)
        if fp is not None:
            sims.append(tanimoto_bits(query_fp, fp))
    return float(1.0 - max(sims)) if sims else 1.0

def load_tables():
    """Load data/ligands_*.csv for TARGETS; skip empty/too-small ones."""
    tables = {}
    for tid in TARGETS:
        f = DATA_DIR / f"ligands_{tid}.csv"
        if f.exists() and f.stat().st_size > 0:
            try:
                df = pd.read_csv(f).dropna(subset=["smiles", "pIC50"]).reset_index(drop=True)
                if len(df) >= 8:
                    tables[tid] = df
            except Exception:
                pass
    return tables
