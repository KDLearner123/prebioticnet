import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_molecule_features(smiles_str: str) -> torch.Tensor:
    """Parses a SMILES string to extract physical and structural vectors."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return torch.zeros(5, dtype=torch.float)
    
    # Feature extraction via RDKit
    mw = Descriptors.MolWt(mol)
    atoms = mol.GetNumAtoms()
    rings = Descriptors.RingCount(mol)
    valence = Descriptors.NumValenceElectrons(mol)
    logp = Descriptors.MolLogP(mol)
    
    return torch.tensor([mw, atoms, rings, valence, logp], dtype=torch.float)

def parse_smiles_reaction(reaction_str: str):
    """
    Parses a reaction SMILES string (Reactants>Agents>Products) 
    into lists of component SMILES.
    """
    if ">" not in reaction_str:
        return [], []
    parts = reaction_str.split(">")
    reactants = [r for r in parts[0].split(".") if r]
    products = [p for p in parts[2].split(".") if p]
    return reactants, products