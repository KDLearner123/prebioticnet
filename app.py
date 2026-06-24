import gradio as gr
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
from src.dataset import build_prebiotic_graph
from src.model import PrebioticGNN
from src.search import find_autocatalytic_loops
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

print("Initializing high-fidelity structural biochemistry GNN...")
graph, molecule_to_idx = build_prebiotic_graph()
idx_to_molecule = {v: k for k, v in molecule_to_idx.items()}

model = PrebioticGNN(mol_in_channels=5, rxn_in_channels=2, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
real_edges = graph.edge_index_dict[('reaction', 'yields', 'molecule')]
num_molecules = graph['molecule'].num_nodes

model.train()
for epoch in range(60):
    optimizer.zero_grad()
    predictions_real = model(graph.x_dict, graph.edge_index_dict, yields_edge_index=real_edges)
    random_molecule_indices = torch.randint(0, num_molecules, (real_edges.shape[1],), device=real_edges.device)
    corrupted_yields_edge = real_edges.clone()
    corrupted_yields_edge[1] = random_molecule_indices
    predictions_fake = model(graph.x_dict, graph.edge_index_dict, yields_edge_index=corrupted_yields_edge)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions_real, torch.ones_like(predictions_real)) + \
           torch.nn.functional.binary_cross_entropy_with_logits(predictions_fake, torch.zeros_like(predictions_fake))
    loss.backward()
    optimizer.step()

model.eval()

def simulate_environment(temperature, ph):
    if temperature < 260 or temperature > 420 or ph < 3.0 or ph > 11.5:
        return f"⚠️ Atmospheric collapse: Extreme conditions disrupted macromolecule chains.", None
        
    temp_deviation = abs(temperature - 298.15) / 100.0
    ph_deviation = abs(ph - 7.0) / 4.0
    search_threshold = min(0.85, max(0.30, 0.50 + (temp_deviation * 0.15) + (ph_deviation * 0.10)))
    
    with torch.no_grad():
        predictions = torch.sigmoid(model(graph.x_dict, graph.edge_index_dict, yields_edge_index=real_edges))
        
    loops = find_autocatalytic_loops(graph, predictions, threshold=search_threshold)
        
    if not loops:
        return f"⚠️ Environmental Shock: Structural affinity insufficient to maintain feedback connectivity.", None
        
    output_text = f"🧬 SUCCESS: Environment Habitable! (System Stress Factor: {search_threshold:.2f})\nDetected {len(loops)} independent loops.\n\n"
    
    # Build an optimized, clean visual network diagram using NetworkX
    # Build an optimized, clean visual network diagram using NetworkX
    plt.figure(figsize=(8, 6)) # Extra margin to handle spacing beautifully
    vis_G = nx.DiGraph()
    
    for idx, loop in enumerate(loops[:2]): # Top 2 loops for clear rendering
        formulas = []
        nodes_list = list(loop)
        for i in range(len(nodes_list)):
            m_idx = int(nodes_list[i].split('_')[1])
            smiles = idx_to_molecule[m_idx]
            
            # --- SMART STRING SANITIZER ---
            rd_m = Chem.MolFromSmiles(smiles)
            if rd_m:
                formula = rdMolDescriptors.CalcMolFormula(rd_m)
            else:
                # If it's a massive raw SMILES string that failed to parse, truncate it cleanly
                if len(smiles) > 10:
                    formula = smiles[:6] + "..."
                else:
                    # Clean up things like "Mol_32" to just a compact identifier "M32"
                    formula = smiles.replace("Mol_", "M")
            
            formulas.append(formula)
            
            # Re-apply the same clean sanitization rules to the target connected node
            next_node = nodes_list[(i + 1) % len(nodes_list)]
            next_m_idx = int(next_node.split('_')[1])
            next_smiles = idx_to_molecule[next_m_idx]
            
            next_rd_m = Chem.MolFromSmiles(next_smiles)
            if next_rd_m:
                next_formula = rdMolDescriptors.CalcMolFormula(next_rd_m)
            else:
                if len(next_smiles) > 10:
                    next_formula = next_smiles[:6] + "..."
                else:
                    next_formula = next_smiles.replace("Mol_", "M")
            
            vis_G.add_edge(formula, next_formula)
            
        trail = " ⇄ ".join(formulas[:4])
        if len(loop) > 4: trail += " ⇄ ... [+ more]"
        output_text += f"Loop #{idx+1} ({len(loop)} molecules):\n   {trail}\n\n"
    
    # --- DE-CLUTTERING VISUAL ENGINE ---
    # 1. Increase k (optimal distance between nodes) and iterations to push elements apart
    pos = nx.spring_layout(vis_G, k=1.2, iterations=50)
    
    # 2. Draw cleanly with distinct colors, slight transparency, and soft borders
    nx.draw_networkx_nodes(vis_G, pos, node_color='#4A90E2', node_size=900, alpha=0.95)
    nx.draw_networkx_edges(vis_G, pos, edgelist=vis_G.edges(), edge_color='#9B9B9B', width=1.5, arrows=True, arrowsize=12)
    
    # 3. Use an adjusted font offset so text doesn't overlap the node circles
    nx.draw_networkx_labels(vis_G, pos, font_size=7, font_weight='bold', font_color='black')
    
    plt.title("Core Autocatalytic Feedback Pathways", fontsize=10, fontweight='bold', pad=10)
    plt.axis('off') # Strip out bounding box border axes
    
    plot_path = "loop_topology.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150) # Increase DPI for sharp text
    plt.close()
    
    return output_text, plot_path
    
    return output_text, plot_path

interface = gr.Interface(
    fn=simulate_environment,
    inputs=[
        gr.Slider(minimum=200, maximum=500, value=298, label="Surface Temperature (Kelvin)"),
        gr.Slider(minimum=1.0, maximum=14.0, value=7.0, label="Atmospheric pH Level")
    ],
    # Added an explicit Image component slot alongside our text console output
    outputs=[
        gr.Textbox(label="Simulation Output Node Metrics", lines=8),
        gr.Image(label="Topological Feedback Loop Mapping")
    ],
    title=" PrebioticNet: In Silico Origin-of-Life Simulator",
    description="Adjust planetary environmental conditions to dynamically evaluate GNN paths and visualize emergent autocatalytic networks."
)

if __name__ == "__main__":
    interface.launch()