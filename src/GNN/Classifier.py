from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as Function

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)                                         #maps input (node_values) -> hidden1 (64) | gather node features + neighbors, normalizes and projects through weight matrix W1
        self.conv2 = GCNConv(hidden_dim, hidden_dim)                                        #maps hidden1 (64) -> hidden2 (64)        | gather node features + neighbors, normalizes and projects through weight matrix W2
        self.classify = nn.Linear(hidden_dim, output_dim)                                   #maps 64 -> classes (classes dataset has)

    def forward(self, data):
        node_features = data.x
        edge_index = data.edge_index
        batch = data.batch                                                   

        node_features = Function.relu(self.conv1(node_features, edge_index))
        node_features = Function.relu(self.conv2(node_features, edge_index))

        node_features = global_mean_pool(node_features, batch)

        return self.classify(node_features)