import torch
from src.dataset import build_prebiotic_graph
from src.model import PrebioticGNN
from src.train import train_prebiotic_model
from src.search import find_autocatalytic_loops
from src.generative import expand_network_with_gnn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors # Added for formula calculation

if __name__ == "__main__":
    print("Compiling raw prebiotic records into Heterogeneous PyG Structures...")
    graph, molecule_to_idx = build_prebiotic_graph()
    idx_to_molecule = {v: k for k, v in molecule_to_idx.items()}
    
    model = PrebioticGNN(mol_in_channels=5, rxn_in_channels=2, hidden_channels=32)
    
    print("\n=== Initializing Optimization Cycle ===")
    trained_model = train_prebiotic_model(model, graph, epochs=100, lr=0.01)
    
    print("\n=== Running Topological Search for Emergent Life Loops ===")
    trained_model.eval()
    with torch.no_grad():
        real_edges = graph.edge_index_dict[('reaction', 'yields', 'molecule')]
        final_predictions = torch.sigmoid(trained_model(graph.x_dict, graph.edge_index_dict, yields_edge_index=real_edges))
        
    loops = find_autocatalytic_loops(graph, final_predictions, threshold=0.70)
    
    print("\n=== SYSTEM ANALYSIS DISCOVERY ===")
    print(f"Identified {len(loops)} independent autocatalytic sub-networks!")
    
    for idx, loop in enumerate(loops[:3]):
        chemical_names = []
        for node in list(loop)[:4]: 
            mol_idx = int(node.split('_')[1])
            smiles = idx_to_molecule[mol_idx]
            
            rd_mol = Chem.MolFromSmiles(smiles)
            # Fixed: Using CalcMolFormula for stable execution
            formula = rdMolDescriptors.CalcMolFormula(rd_mol) if rd_mol else smiles
            chemical_names.append(formula)
            
        trail = " ⇄ ".join(chemical_names)
        if len(loop) > 4:
            trail += " ⇄ ... [+ more]"
        print(f"Loop #{idx+1} Formula Flow: {trail}")

    # Run the expansion engine
    expand_network_with_gnn(trained_model, graph, molecule_to_idx, threshold=0.50)