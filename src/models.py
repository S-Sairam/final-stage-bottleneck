# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=1, num_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class GNN(nn.Module):
    """ A generalized GNN that can be instantiated with different layer types (e.g., GCN, GAT). """
    def __init__(self, in_dim, hidden_dim=32, out_dim=1, num_layers=2, layer_type='gcn', heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.layer_type = layer_type.lower()
        
        GNNLayer = GCNConv
        if self.layer_type == 'gat':
            GNNLayer = GATConv
        
        # First layer
        in_channels_next = in_dim
        if self.layer_type == 'gat':
            self.convs.append(GNNLayer(in_channels_next, hidden_dim, heads=heads))
            in_channels_next = hidden_dim * heads
        else:
            self.convs.append(GNNLayer(in_channels_next, hidden_dim))
            in_channels_next = hidden_dim

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GNNLayer(in_channels_next, hidden_dim))
            in_channels_next = hidden_dim
            
        # Final layer
        if self.layer_type == 'gat':
            self.convs.append(GNNLayer(in_channels_next, out_dim, heads=1))
        else:
            self.convs.append(GNNLayer(in_channels_next, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class FinalGNN(nn.Module):
    """ The GNN model for the final-stage R-Learner, now architecturally flexible. """
    def __init__(self, in_dim, **model_kwargs):
        super().__init__()
        self.gnn = GNN(in_dim, **model_kwargs)
    def forward(self, x, edge_index): return self.gnn(x, edge_index)

class T_Learner_GNN(nn.Module):
    """ The GNN model for the T-Learner baseline, now architecturally flexible. """
    def __init__(self, in_dim, **model_kwargs):
        super().__init__()
        self.gnn = GNN(in_dim + 1, **model_kwargs)
    def forward(self, x, t, edge_index):
        xt = torch.cat([x, t], dim=1)
        return self.gnn(xt, edge_index)
