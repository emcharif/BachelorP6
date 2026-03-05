from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as Function
import torch

class TemporalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps=20):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)                 #mapper input (11) -> hidden1 (64) | aggregater nodens egens features + naboer, derefter projekterer gennem en weight matrix1
        self.conv2 = GCNConv(hidden_dim, hidden_dim)                #mapper hidden1 (64) -> hidden2 (64) | aggregater nodens egens features + naboer, derefter projekterer gennem en weight matrix2
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True) #input 64 dimensional vectors, outputs 64 dimensional hidden state, shape: [batch, timestep, features]
        self.classify = nn.Linear(hidden_dim, output_dim)           #mapper 64 -> 3 (classes)

    def forward(self, data):
        vehicle_x  = data.vehicle.x
        edge_index = data[('vehicle', 'to', 'vehicle')].edge_index

        h = Function.relu(self.conv1(vehicle_x, edge_index))        #pass all vehicle nodes through hidden layer 1, shape: [vehicle nodes, 64] | each node aggregates attributes from neighbors via edges
        h = Function.relu(self.conv2(h, edge_index))                #does it again, but with hidden layer 2

        ptr   = data.vehicle.ptr                                    #indicates where each timesteps node start and end, shape: [0,0,0,1,1,2,2]
        vehicle_nodes_pr_timestep = ptr[1:] - ptr[:-1]              #indicates how many vehicle nodes pr timestep, size: 20 -> cause 20 timesteps
        timestep_batch = torch.repeat_interleave(torch.arange(len(vehicle_nodes_pr_timestep), device=vehicle_x.device), vehicle_nodes_pr_timestep) #expander index by size, if timestep 0 has 3 vehicle nodes og timestep 1 has 4 vehicle nodes -> [0,0,0,1,1,1,1]

        h_pooled = global_mean_pool(h, timestep_batch)              #tager average af hver timesteps vehicle nodes og mapper til 64 dimensions, shape bliver altså: [20, 64] 20 = timesteps
        h_seq    = h_pooled.unsqueeze(0)                            #konverterer bare h_pooled om til [batch, timestep, features] fordi det er hvad GRU forventer at få

        _, h_final = self.gru(h_seq)                                #her processer GRU de 20 timesteps med de 64 dimensioner, og opdaterer det næste timesteps "hidden memory"
        h_final = h_final.squeeze(0)                                #vi gemmer den sidste hidden, fordi det er svaret. Vi squeezer så vi får [1, 3], som betyder de 3 klasser. Det vil sige den med højest value er prediction

        return self.classify(h_final)            