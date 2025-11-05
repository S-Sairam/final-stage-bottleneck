# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
    def __init__(self, in_dim, hidden_dim=32, out_dim=1, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

class FinalGNN(nn.Module):
    """ The GNN model for the final-stage R-Learner. """
    def __init__(self, in_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.gnn = GNN(in_dim, hidden_dim, out_dim=1, num_layers=num_layers)
    def forward(self, x, edge_index): return self.gnn(x, edge_index)

class T_Learner_GNN(nn.Module):
    """ The GNN model for the T-Learner baseline. Takes features and treatment as input. """
    def __init__(self, in_dim, hidden_dim=32, out_dim=1, num_layers=2):
        super().__init__()
        # Input dimension is feature dim + 1 for the treatment variable
        self.gnn = GNN(in_dim + 1, hidden_dim, out_dim, num_layers)
    def forward(self, x, t, edge_index):
        xt = torch.cat([x, t], dim=1)
        return self.gnn(xt, edge_index)
