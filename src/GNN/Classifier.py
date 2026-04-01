from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as Function
import torch

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)                                         #maps input (node_values) -> hidden1 (64) | gather node features + neighbors, normalizes and projects through weight matrix W1
        self.conv2 = GCNConv(hidden_dim, hidden_dim)                                        #maps hidden1 (64) -> hidden2 (64)        | gather node features + neighbors, normalizes and projects through weight matrix W2
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first = True)
        self.classify = nn.Linear(hidden_dim, output_dim)                                   #maps 64 -> classes (classes dataset has)

    def forward(self, data):
        vehicle_node = data.vehicle.x
        edge_index = data[('vehicle', 'to', 'vehicle')].edge_index                                                

        h = Function.relu(self.conv1(vehicle_node, edge_index))
        h = Function.relu(self.conv2(h, edge_index))

        timestep_start_end = data.vehicle.ptr
        vehicle_nodes_pr_timestep = timestep_start_end[1:] - timestep_start_end[:-1]
        timestep_batch = torch.repeat_interleave(torch.arange(len(vehicle_nodes_pr_timestep), device = vehicle_node.device), vehicle_nodes_pr_timestep)

        h_pooled = global_mean_pool(h, timestep_batch)
        h_seq = h_pooled.unsqueeze(0)

        _, h_final = self.gru(h_seq)
        h_final = h_final.squeeze(0)

        return self.classify(h_final)