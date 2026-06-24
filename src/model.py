import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class PrebioticGNN(torch.nn.Module):
    def __init__(self, mol_in_channels, rxn_in_channels, hidden_channels):
        super().__init__()
        
        # Message passing layers
        self.conv1 = HeteroConv({
            ('molecule', 'participates_in', 'reaction'): SAGEConv((mol_in_channels, rxn_in_channels), hidden_channels),
            ('reaction', 'yields', 'molecule'): SAGEConv((rxn_in_channels, mol_in_channels), hidden_channels),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('molecule', 'participates_in', 'reaction'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('reaction', 'yields', 'molecule'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
        }, aggr='sum')
        
        # Link Predictor MLPs: Takes (Reaction hidden state + Molecule hidden state) -> Score
        # We predict feasibility based on whether a specific reaction actually yields a specific product
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict, yields_edge_index=None):
        # 1. Propagate structural messages across the network
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # 2. Extract edge configurations
        if yields_edge_index is None:
            yields_edge_index = edge_index_dict[('reaction', 'yields', 'molecule')]
            
        rxn_indices = yields_edge_index[0]
        mol_indices = yields_edge_index[1]
        
        # Gather representations corresponding to the specific edges evaluated
        rxn_features = x_dict['reaction'][rxn_indices]
        mol_features = x_dict['molecule'][mol_indices]
        
        # Concatenate edge traits to calculate unique link compatibility
        edge_embeddings = torch.cat([rxn_features, mol_features], dim=-1)
        
        return self.link_predictor(edge_embeddings)