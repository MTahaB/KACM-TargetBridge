import hashlib
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import RDLogger
from src.utils.chem import murcko_scaffold_smiles

RDLogger.DisableLog("rdApp.*")


def _stable_noscaffold_id(smiles: str) -> str:
    digest = hashlib.sha1(str(smiles).encode("utf-8")).hexdigest()[:12]
    return f"NOSCAF_{digest}"


def scaffold_split(
    df: pd.DataFrame, smiles_col="smiles", test_frac=0.2, seed=42
) -> Tuple[np.ndarray, np.ndarray]:
    scaffolds = df[smiles_col].map(
        lambda s: murcko_scaffold_smiles(s) or _stable_noscaffold_id(s)
    )
    groups = scaffolds.groupby(scaffolds).indices  # dict scaffold -> indices
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n = len(df)
    test_target = int(round(test_frac * n))
    test_idx, acc = [], 0
    for k in keys:
        idxs = list(groups[k])
        test_idx.extend(idxs)
        acc += len(idxs)
        if acc >= test_target:
            break
    test_mask = np.zeros(n, dtype=bool)
    test_mask[np.array(test_idx, dtype=int)] = True
    train_mask = ~test_mask
    return train_mask, test_mask
