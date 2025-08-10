from typing import Optional
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol

def canonical_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if not mol:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True)
