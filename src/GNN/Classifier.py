import torch.nn as nn
import torch.nn.functional as Function
import torch

from torch_geometric.nn import GINConv, global_max_pool

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2)

        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv3 = GINConv(nn3)
        self.classify = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x1 = Function.relu(self.conv1(x, edge_index))
        x2 = Function.relu(self.conv2(x1, edge_index))
        x3 = Function.relu(self.conv3(x2, edge_index))

        out = torch.cat([
            global_max_pool(x1, batch),
            global_max_pool(x2, batch),
            global_max_pool(x3, batch),
        ], dim=1)

        out = Function.dropout(out, p=0.5, training=self.training)
        return self.classify(out)