import torch
import torch.nn as nn
import torch.nn.functional as Function

from torch_geometric.nn import GINConv, global_add_pool


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINConv(nn2)

        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv3 = GINConv(nn3)

        graph_embedding_dim = hidden_dim * 3

        self.classify = nn.Linear(graph_embedding_dim, output_dim)

        self.watermark_head = nn.Sequential(
            nn.Linear(graph_embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, data, return_watermark_score: bool = False):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x1 = Function.relu(self.conv1(x, edge_index))
        x2 = Function.relu(self.conv2(x1, edge_index))
        x3 = Function.relu(self.conv3(x2, edge_index))

        graph_embedding = torch.cat(
            [
                global_add_pool(x1, batch),
                global_add_pool(x2, batch),
                global_add_pool(x3, batch),
            ],
            dim=1,
        )

        class_logits = self.classify(graph_embedding)

        if return_watermark_score:
            watermark_score = self.watermark_head(graph_embedding)
            return class_logits, watermark_score

        return class_logits