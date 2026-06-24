import os
import json
import torch
from torch_geometric.data import HeteroData
from src.config import BASE_DIR
from src.features import get_molecule_features, parse_smiles_reaction

def build_prebiotic_graph() -> HeteroData:
    data = HeteroData()
    
    molecule_to_idx = {}
    molecule_features = []
    reaction_features = []
    edge_reactant_to_rxn = []
    edge_rxn_to_product = []
    rxn_counter = 0

    # Fallback search: look in 'conditions', then 'networks', then the root data folder
    possible_paths = [
        os.path.join(BASE_DIR, "data", "chemorigins-data", "conditions"),
        os.path.join(BASE_DIR, "data", "chemorigins-data", "networks"),
        os.path.join(BASE_DIR, "data", "chemorigins-data")
    ]
    
    target_dir = None
    for path in possible_paths:
        if os.path.exists(path) and any(f.endswith('.json') for f in os.listdir(path)):
            target_dir = path
            break
            
    if target_dir is None:
        raise FileNotFoundError(
            f"Could not find any directory containing prebiotic JSON records in 'data/chemorigins-data/'. "
            f"Please verify what folders exist inside that path."
        )
        
    print(f"Parsing prebiotic files from target directory: {target_dir}")

    for file_name in os.listdir(target_dir):
        if not file_name.endswith(".json"):
            continue
            
        with open(os.path.join(target_dir, file_name), 'r') as f:
            try:
                entry = json.load(f)
            except json.JSONDecodeError:
                continue
                
            # Fallback for different JSON keys used in prebiotic chemistry graphs
            # Some databases use 'reaction', others use 'equation' or 'smiles'
            rxn_smiles = entry.get("reaction", entry.get("equation", entry.get("smiles", "")))
            reactants, products = parse_smiles_reaction(rxn_smiles)
            
            if not reactants or not products:
                continue
                
            temp = float(entry.get("temperature", entry.get("t", 298.15)))
            ph = float(entry.get("ph", 7.0))
            
            reaction_features.append([temp, ph])
            
            for r_smiles in reactants:
                if r_smiles not in molecule_to_idx:
                    molecule_to_idx[r_smiles] = len(molecule_to_idx)
                    molecule_features.append(get_molecule_features(r_smiles))
                edge_reactant_to_rxn.append([molecule_to_idx[r_smiles], rxn_counter])
                
            for p_smiles in products:
                if p_smiles not in molecule_to_idx:
                    molecule_to_idx[p_smiles] = len(molecule_to_idx)
                    molecule_features.append(get_molecule_features(p_smiles))
                edge_rxn_to_product.append([rxn_counter, molecule_to_idx[p_smiles]])
                
            rxn_counter += 1

    # Check to prevent the TensorList empty error before it happens
    if len(molecule_features) == 0:
        raise ValueError(
            "Found JSON files, but couldn't parse any valid reaction strings. "
            "The JSON structure might be different than expected."
        )

    data['molecule'].x = torch.stack(molecule_features)
    data['reaction'].x = torch.tensor(reaction_features, dtype=torch.float)
    
    data['molecule', 'participates_in', 'reaction'].edge_index = torch.tensor(edge_reactant_to_rxn, dtype=torch.long).t().contiguous()
    data['reaction', 'yields', 'molecule'].edge_index = torch.tensor(edge_rxn_to_product, dtype=torch.long).t().contiguous()
    
    return data, molecule_to_idx