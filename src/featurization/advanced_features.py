import numpy as np
from typing import Optional, Union, List, Tuple
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from src.utils.chem import mol_from_smiles

def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> Optional[np.ndarray]:
    """Enhanced Morgan fingerprints with feature invariants."""
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    
    # Use feature invariants for better chemical representation
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, 
        useFeatures=True,  # Use pharmacophore features
        useChirality=True  # Include chirality information
    )
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def rdkit_2d_descriptors(smiles: str) -> Optional[np.ndarray]:
    """Comprehensive RDKit 2D molecular descriptors."""
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    
    descriptors = []
    
    # Basic molecular properties
    descriptors.extend([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCsp3(mol),
    ])
    
    # Lipinski descriptors
    descriptors.extend([
        Descriptors.qed(mol),  # Drug-likeness
        rdMolDescriptors.BertzCT(mol),  # Complexity
        rdMolDescriptors.BalabanJ(mol),  # Balaban index
        rdMolDescriptors.Kappa1(mol),   # Shape indices
        rdMolDescriptors.Kappa2(mol),
        rdMolDescriptors.Kappa3(mol),
    ])
    
    # Graph-based descriptors
    descriptors.extend([
        rdMolDescriptors.Chi0v(mol),    # Connectivity indices
        rdMolDescriptors.Chi1v(mol),
        rdMolDescriptors.Chi2v(mol),
        rdMolDescriptors.Chi3v(mol),
        rdMolDescriptors.Chi4v(mol),
        rdMolDescriptors.HallKierAlpha(mol),
    ])
    
    # Electronic descriptors
    try:
        descriptors.extend([
            rdMolDescriptors.MaxEStateIndex(mol),
            rdMolDescriptors.MinEStateIndex(mol),
            rdMolDescriptors.MaxAbsEStateIndex(mol),
            rdMolDescriptors.MinAbsEStateIndex(mol),
        ])
    except:
        descriptors.extend([0.0, 0.0, 0.0, 0.0])
    
    return np.array(descriptors, dtype=np.float32)

def pharmacophore_fingerprint(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Pharmacophore-based fingerprint using Gobbi features."""
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    
    try:
        factory = Gobbi_Pharm2D.factory
        fp = Generate.Gen2DFingerprint(mol, factory, dMat=Chem.GetDistanceMatrix(mol))
        
        # Convert to bit vector
        bit_fp = DataStructs.CreateFromBitString("".join([str(x) for x in fp]))
        arr = np.zeros((n_bits,), dtype=np.uint8)
        
        # Fold to desired length if necessary
        if fp.GetNumBits() <= n_bits:
            DataStructs.ConvertToNumpyArray(bit_fp, arr[:fp.GetNumBits()])
        else:
            # Fold fingerprint
            folded = DataStructs.FoldFingerprint(bit_fp, n_bits)
            DataStructs.ConvertToNumpyArray(folded, arr)
        
        return arr
    except:
        # Fallback to zeros if pharmacophore generation fails
        return np.zeros(n_bits, dtype=np.uint8)

def atom_pair_fingerprint(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Atom pair fingerprint for structural similarity."""
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    
    fp = Pairs.GetAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def topological_torsion_fp(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Topological torsion fingerprint."""
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def layered_fingerprint(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Layered fingerprint with substructure patterns."""
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    
    fp = rdMolDescriptors.LayeredFingerprint(mol, fpSize=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def combined_molecular_features(smiles: str, feature_types: List[str] = None, 
                               fp_bits: int = 1024) -> Optional[np.ndarray]:
    """
    Combine multiple molecular representations into a single feature vector.
    
    Args:
        smiles: SMILES string
        feature_types: List of feature types to include
        fp_bits: Number of bits for each fingerprint type
        
    Returns:
        Combined feature vector or None if molecule is invalid
    """
    if feature_types is None:
        feature_types = ['morgan', 'descriptors', 'atom_pairs', 'topological']
    
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    
    features = []
    
    for ft in feature_types:
        if ft == 'morgan':
            fp = morgan_fp(smiles, n_bits=fp_bits)
            if fp is not None:
                features.append(fp)
        
        elif ft == 'descriptors':
            desc = rdkit_2d_descriptors(smiles)
            if desc is not None:
                # Normalize descriptors to [0,1] range roughly
                desc_norm = np.clip(desc / (np.abs(desc) + 1e-6), -5, 5)
                features.append(desc_norm)
        
        elif ft == 'pharmacophore':
            fp = pharmacophore_fingerprint(smiles, n_bits=fp_bits)
            if fp is not None:
                features.append(fp)
        
        elif ft == 'atom_pairs':
            fp = atom_pair_fingerprint(smiles, n_bits=fp_bits)
            if fp is not None:
                features.append(fp)
        
        elif ft == 'topological':
            fp = topological_torsion_fp(smiles, n_bits=fp_bits)
            if fp is not None:
                features.append(fp)
        
        elif ft == 'layered':
            fp = layered_fingerprint(smiles, n_bits=fp_bits)
            if fp is not None:
                features.append(fp)
    
    if not features:
        return None
    
    # Concatenate all features
    combined = np.concatenate(features).astype(np.float32)
    return combined

def featurize_smiles_list_advanced(smiles_list: List[str], 
                                 feature_types: List[str] = None,
                                 fp_bits: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced featurization with multiple molecular representations.
    
    Returns:
        X: Feature matrix (n_valid_molecules, n_features)
        keep_idx: Indices of valid molecules
    """
    if feature_types is None:
        feature_types = ['morgan', 'descriptors', 'atom_pairs']
    
    X, keep_idx = [], []
    
    for i, smiles in enumerate(smiles_list):
        features = combined_molecular_features(smiles, feature_types, fp_bits)
        if features is not None:
            X.append(features)
            keep_idx.append(i)
    
    if not X:
        # Return empty arrays with appropriate shape
        dummy_features = combined_molecular_features("C", feature_types, fp_bits)
        n_features = len(dummy_features) if dummy_features is not None else fp_bits
        return np.empty((0, n_features), dtype=np.float32), np.array([], dtype=int)
    
    return np.array(X, dtype=np.float32), np.array(keep_idx, dtype=int)

def get_feature_names(feature_types: List[str] = None, fp_bits: int = 1024) -> List[str]:
    """Get feature names for interpretability."""
    if feature_types is None:
        feature_types = ['morgan', 'descriptors', 'atom_pairs']
    
    names = []
    
    for ft in feature_types:
        if ft == 'morgan':
            names.extend([f"morgan_bit_{i}" for i in range(fp_bits)])
        elif ft == 'descriptors':
            desc_names = [
                'mol_wt', 'mol_logp', 'h_donors', 'h_acceptors', 'tpsa',
                'rot_bonds', 'arom_rings', 'sat_rings', 'ring_count', 'frac_csp3',
                'qed', 'bertz_ct', 'balaban_j', 'kappa1', 'kappa2', 'kappa3',
                'chi0v', 'chi1v', 'chi2v', 'chi3v', 'chi4v', 'hall_kier_alpha',
                'max_estate', 'min_estate', 'max_abs_estate', 'min_abs_estate'
            ]
            names.extend(desc_names)
        elif ft in ['pharmacophore', 'atom_pairs', 'topological', 'layered']:
            names.extend([f"{ft}_bit_{i}" for i in range(fp_bits)])
    
    return names
