"""Molecular featurization utilities.

This module keeps Morgan fingerprints cached on disk because the same SMILES are
often featurized repeatedly during Streamlit prediction, benchmarking, and model
training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from joblib import Memory
from rdkit import DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

from src.utils.chem import mol_from_smiles

RDLogger.DisableLog("rdApp.*")

CACHE_DIR = Path(os.environ.get("TARGETBRIDGE_CACHE_DIR", ".cache/joblib"))
memory = Memory(location=CACHE_DIR, verbose=0)


@memory.cache
def _cached_morgan_fp(smiles: str, n_bits: int, radius: int) -> np.ndarray | None:
    """Compute a Morgan fingerprint for a single molecule.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    n_bits:
        Fingerprint length.
    radius:
        Morgan fingerprint radius.

    Returns
    -------
    numpy.ndarray or None
        Binary fingerprint as uint8, or None if the molecule is invalid.
    """
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    """Return a cached Morgan fingerprint.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    n_bits:
        Fingerprint length.
    radius:
        Morgan fingerprint radius.

    Returns
    -------
    numpy.ndarray or None
        Binary fingerprint as uint8, or None if the molecule is invalid.
    """
    fp = _cached_morgan_fp(smiles, n_bits, radius)
    return None if fp is None else fp.copy()


def rdkit_descriptor_vector(smiles: str) -> np.ndarray | None:
    """Compute compact RDKit physicochemical descriptors.

    Parameters
    ----------
    smiles:
        Input SMILES string.

    Returns
    -------
    numpy.ndarray or None
        Descriptor vector containing MW, logP, TPSA, HBD, HBA, rotatable bonds,
        aromatic rings, ring count, and fraction sp3.
    """
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
        ],
        dtype=np.float32,
    )


def morgan_plus_rdkit_features(
    smiles: str, n_bits: int = 2048, radius: int = 2
) -> np.ndarray | None:
    """Concatenate Morgan bits and scaled RDKit descriptors.

    Descriptor values are divided by simple constants to keep them in a range
    closer to binary fingerprint values for tree and linear models.
    """
    fp = morgan_fp(smiles, n_bits=n_bits, radius=radius)
    desc = rdkit_descriptor_vector(smiles)
    if fp is None or desc is None:
        return None
    scale = np.array([500.0, 5.0, 150.0, 5.0, 10.0, 10.0, 5.0, 8.0, 1.0])
    desc_scaled = np.nan_to_num(desc / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([fp.astype(np.float32), desc_scaled.astype(np.float32)])


def featurize_smiles_list(
    smiles_list: Iterable[str],
    n_bits: int = 2048,
    radius: int = 2,
    feature_set: str = "morgan",
) -> tuple[np.ndarray, np.ndarray]:
    """Featurize a sequence of SMILES.

    Parameters
    ----------
    smiles_list:
        Iterable of SMILES strings.
    n_bits:
        Morgan fingerprint length.
    radius:
        Morgan fingerprint radius.
    feature_set:
        Either ``"morgan"`` or ``"morgan_rdkit"``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Feature matrix and indices of valid molecules.
    """
    X, keep_idx = [], []
    for i, smiles in enumerate(smiles_list):
        if feature_set == "morgan":
            vec = morgan_fp(smiles, n_bits=n_bits, radius=radius)
        elif feature_set == "morgan_rdkit":
            vec = morgan_plus_rdkit_features(smiles, n_bits=n_bits, radius=radius)
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        if vec is not None:
            X.append(vec)
            keep_idx.append(i)

    dtype = np.uint8 if feature_set == "morgan" else np.float32
    return np.array(X, dtype=dtype), np.array(keep_idx, dtype=int)
