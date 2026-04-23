import torch.nn as nn
import torch.nn.functional as Function
import torch

from torch_geometric.nn import GINConv, global_add_pool


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

        # ── Classification head (unchanged) ───────────────────────────────
        self.classify = nn.Linear(hidden_dim * 3, output_dim)

        # ── Watermark detection head ───────────────────────────────────────
        # Sits on top of the same pooled graph representation as the
        # classifier. Trained to output 1.0 for watermarked graphs and
        # 0.0 for clean graphs — learns the chain topology signal directly
        # from the GNN's structural embeddings, not from feature values.
        # A small bottleneck (hidden_dim // 2) keeps it lightweight so it
        # doesn't interfere with the classification head's gradients.
        self.watermark_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()   # output in [0, 1]: 1.0 = watermarked, 0.0 = clean
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x1 = Function.relu(self.conv1(x, edge_index))
        x2 = Function.relu(self.conv2(x1, edge_index))
        x3 = Function.relu(self.conv3(x2, edge_index))

        out = torch.cat([
            global_add_pool(x1, batch),
            global_add_pool(x2, batch),
            global_add_pool(x3, batch),
        ], dim=1)

        # Both heads share the same graph-level representation
        class_logits = self.classify(out)
        watermark_score = self.watermark_head(out)   # shape: [batch_size, 1]

        return class_logits, watermark_score