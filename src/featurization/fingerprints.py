import numpy as np
from typing import Union, Optional
from rdkit import DataStructs, RDLogger
from rdkit.Chem import AllChem
from src.utils.chem import mol_from_smiles

RDLogger.DisableLog("rdApp.*")

def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def featurize_smiles_list(smiles_list, n_bits=2048, radius=2):
    X, keep_idx = [], []
    for i, s in enumerate(smiles_list):
        vec = morgan_fp(s, n_bits=n_bits, radius=radius)
        if vec is not None:
            X.append(vec)
            keep_idx.append(i)
    return (np.array(X, dtype=np.uint8), np.array(keep_idx, dtype=int))
