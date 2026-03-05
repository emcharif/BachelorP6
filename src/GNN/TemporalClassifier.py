from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as Function
import torch

class TemporalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps=20):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classify = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        vehicle_x  = data.vehicle.x
        edge_index = data[('vehicle', 'to', 'vehicle')].edge_index

        h = Function.relu(self.conv1(vehicle_x, edge_index))
        h = Function.relu(self.conv2(h, edge_index))

        # ptr is intact when graphs are not collated by PyG
        # e.g. [0, 12, 24, ..., 240] for 20 timesteps
        ptr   = data.vehicle.ptr
        sizes = ptr[1:] - ptr[:-1]
        timestep_batch = torch.repeat_interleave(torch.arange(len(sizes), device=vehicle_x.device), sizes)

        h_pooled = global_mean_pool(h, timestep_batch)  # [20, hidden_dim]
        h_seq    = h_pooled.unsqueeze(0)                # [1, 20, hidden_dim]

        _, h_final = self.gru(h_seq)
        h_final = h_final.squeeze(0)                    # [1, hidden_dim]

        return self.classify(h_final)                   # [1, output_dim]