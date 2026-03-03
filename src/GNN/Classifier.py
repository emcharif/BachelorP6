from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as Function

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.vehicle.x
        edge_index = data[('vehicle', 'to', 'vehicle')].edge_index
        batch = data.vehicle.batch
    
        h = Function.relu(self.conv1(x, edge_index)) #activation function
        h = Function.relu(self.conv2(h, edge_index)) #activation function
    
        hg = global_mean_pool(h, batch)
        return self.classify(hg)