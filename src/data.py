# src/data.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import erdos_renyi_graph, stochastic_blockmodel_graph, barabasi_albert_graph
from torch_geometric.datasets import Planetoid

# We need a GNN definition here for the simulation, so we define a simple one.
# This avoids a circular import from src.models.
class SimGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def simulate_data(seed=42, n=1000, d=10, graph_type='ba', cate_type='simple_h', real_data_name=None):
    """
    Simulates a world with configurable graph topology and CATE function.
    Can operate in fully-synthetic or semi-synthetic mode.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    
    if real_data_name:
        try:
            dataset = Planetoid(root='/tmp/Cora', name='Cora')
            real_data = dataset[0]
            X, edge_index = real_data.x, real_data.edge_index
            n, d = X.shape
        except Exception as e:
            print(f"ERROR: Could not load real dataset '{real_data_name}'. {e}")
            return None
    else:
        X = torch.randn(n, d)
        if graph_type == 'er':
            edge_index = erdos_renyi_graph(n, edge_prob=0.05)
        elif graph_type == 'sbm':
            block_sizes = torch.tensor([n // 2, n - (n // 2)])
            edge_probs = torch.tensor([[0.1, 0.01], [0.01, 0.1]])
            edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
        else:
            edge_index = barabasi_albert_graph(n, num_edges=4)
        
    confounder_gnn_1 = SimGNN(d, d, d)
    confounder_gnn_2 = SimGNN(d, d, d)
    with torch.no_grad():
        H_1hop = F.relu(confounder_gnn_1(X, edge_index))
        H_2hop = F.relu(confounder_gnn_2(H_1hop, edge_index))

    T_logits = 0.5 * X[:, 0] - 0.7 * H_1hop[:, 1] + 0.3 * torch.randn(n)
    T = (torch.sigmoid(T_logits) > 0.5).float().unsqueeze(1)

    if cate_type == 'local_x':
        true_causal_effect = 2.0 + 1.5 * torch.sin(X[:, 0])
    elif cate_type == 'higher_order':
        true_causal_effect = 2.0 + 1.5 * torch.sin(H_2hop[:, 0])
    elif cate_type == 'interaction':
        true_causal_effect = 2.0 + 1.0 * (H_1hop[:, 0] * X[:, 1])
    else:
        true_causal_effect = 2.0 + 1.5 * torch.sin(H_1hop[:, 0])
    
    Y = (H_1hop[:, 0] + 0.5 * X[:, 1] + 0.5 * torch.randn(n) + 
         T.squeeze() * true_causal_effect)
    
    return X, T, Y.unsqueeze(1), edge_index, true_causal_effect
