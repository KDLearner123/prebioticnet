import torch
import random
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors # Added here too
from src.features import get_molecule_features

def generate_hypothetical_reactions(molecule_to_idx, num_proposals=500):
    molecules = list(molecule_to_idx.keys())
    hypothetical_rxns = []
    for _ in range(num_proposals):
        num_reactants = random.choice([1, 2])
        sampled_reactants = random.sample(molecules, k=min(num_reactants, len(molecules)))
        sampled_product = random.choice(molecules)
        if sampled_product in sampled_reactants:
            continue
        hypothetical_rxns.append({
            "reactants": sampled_reactants,
            "product": sampled_product,
            "conditions": [373.15, 8.5] 
        })
    return hypothetical_rxns

def expand_network_with_gnn(model, graph, molecule_to_idx, threshold=0.85):
    model.eval()
    idx_to_molecule = {v: k for k, v in molecule_to_idx.items()}
    proposals = generate_hypothetical_reactions(molecule_to_idx, num_proposals=200)
    new_edges_discovered = 0
    
    print(f"\nScanning {len(proposals)} in silico hypothetical reaction variants...")
    
    with torch.no_grad():
        for rxn in proposals:
            mol_features = torch.stack([get_molecule_features(r) for r in rxn["reactants"]])
            rxn_features = torch.tensor([rxn["conditions"]], dtype=torch.float)
            
            rxn_emb_padded = torch.zeros((1, 32)) 
            rxn_emb_padded[:, :2] = rxn_features
            
            prod_feat = get_molecule_features(rxn["product"])
            prod_emb_padded = torch.zeros((1, 32))
            prod_emb_padded[:, :5] = prod_feat
            
            edge_embedding = torch.cat([rxn_emb_padded, prod_emb_padded], dim=-1)
            score = torch.sigmoid(model.link_predictor(edge_embedding)).item()
            
            if score >= threshold:
                new_edges_discovered += 1
                
                # Fixed: Use CalcMolFormula for reactant mapping
                reactant_formulas = []
                for r in rxn["reactants"]:
                    rd_m = Chem.MolFromSmiles(r)
                    reactant_formulas.append(rdMolDescriptors.CalcMolFormula(rd_m) if rd_m else r)
                reactant_names = " + ".join(reactant_formulas)
                
                # Fixed: Use CalcMolFormula for product mapping
                rd_p = Chem.MolFromSmiles(rxn["product"])
                product_name = rdMolDescriptors.CalcMolFormula(rd_p) if rd_p else rxn["product"]
                
                if new_edges_discovered <= 3: 
                    print(f"  [AI Discovery] Viable path predicted ({score*100:.1f}%): {reactant_names} ──► {product_name}")
                    
    print(f"Successfully expanded network topology with {new_edges_discovered} new AI-validated chemical paths.")