import networkx as nx

def find_autocatalytic_loops(graph, predictions, threshold=0.75):
    """
    Traverses the GNN's highly confident predictions to extract closed,
    self-sustaining reaction feedback loops.
    """
    # Create a directed graph for topology analysis
    G = nx.DiGraph()
    
    # Get the mapping arrays of our yields edges
    yields_edges = graph.edge_index_dict[('reaction', 'yields', 'molecule')]
    participates_edges = graph.edge_index_dict[('molecule', 'participates_in', 'reaction')]
    
    # Convert participates to a fast lookup dictionary: reaction_idx -> list of reactant molecules
    rxn_to_reactants = {}
    for i in range(participates_edges.shape[1]):
        mol_idx = participates_edges[0, i].item()
        rxn_idx = participates_edges[1, i].item()
        if rxn_idx not in rxn_to_reactants:
            rxn_to_reactants[rxn_idx] = []
        rxn_to_reactants[rxn_idx].append(mol_idx)

    # Step 1: Populate edges where the GNN validates the path confidence
    for i in range(yields_edges.shape[1]):
        rxn_idx = yields_edges[0, i].item()
        prod_idx = yields_edges[1, i].item()
        score = predictions[i].item()
        
        # If the GNN says this reaction is highly feasible
        if score >= threshold:
            reactants = rxn_to_reactants.get(rxn_idx, [])
            for r_idx in reactants:
                # Directed Edge: Reactant Molecule -> Product Molecule via a valid reaction path
                G.add_edge(f"Mol_{r_idx}", f"Mol_{prod_idx}", weight=score)
                
    # Step 2: Use Tarjan's algorithm to find Strongly Connected Components (loops)
    all_loops = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    
    return all_loops