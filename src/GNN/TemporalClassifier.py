from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as Function
import torch

class TemporalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)                                         #mapper input (11) -> hidden1 (64) | samler nodens egens features + naboers features og normaliserer, derefter projekterer gennem en weight matrix1
        self.conv2 = GCNConv(hidden_dim, hidden_dim)                                        #mapper hidden1 (64) -> hidden2 (64) | samler nodens egens features + naboers features og normaliserer, derefter projekterer gennem en weight matrix2
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)                         #input 64 dimensional vectors, outputs 64 dimensional hidden state, form: [batch, timestep, features]
        self.classify = nn.Linear(hidden_dim, output_dim)                                   #mapper 64 -> 3 (de 3 klasser vi har)

    def forward(self, data):
        vehicle_x  = data.vehicle.x
        edge_index = data[('vehicle', 'to', 'vehicle')].edge_index

        hidden_layer = Function.relu(self.conv1(vehicle_x, edge_index))                     #kører alle vehicles nodes gennem hidden layer1, form: [vehicle nodes, 64] | hver node får værdier fra dens naboer via edges
        hidden_layer = Function.relu(self.conv2(hidden_layer, edge_index))                  #gør det samme med outut af hidden layer1 og kører igennem hidden layer2

        timestep_start_end = data.vehicle.ptr                                               #indikerer hvor hvert timesteps node starter og slutter, form: [0, 8, 14...149]
        vehicle_nodes_pr_timestep = timestep_start_end[1:] - timestep_start_end[:-1]        #indikerer hvor mange vehicle nodes der er pr timestep, størrelse: 20
        timestep_batch = torch.repeat_interleave(torch.arange(len(vehicle_nodes_pr_timestep), device=vehicle_x.device), vehicle_nodes_pr_timestep) #mapper hvert vehicle node til dets timestep, altså hvis timestep 0 har 3 vehicle nodes, så [0,0,0, 1,1] så har timestep 1 2 vehicle nodes

        h_pooled = global_mean_pool(hidden_layer, timestep_batch)                           #tager gennemsnittet af hver timesteps vehicle nodes og mapper til 64 dimensions, form bliver altså: [20, 64] 20 = timesteps
        h_seq    = h_pooled.unsqueeze(0)                                                    #konverterer bare h_pooled om til [batch, timestep, features] fordi det er hvad GRU forventer at få

        _, h_final = self.gru(h_seq)                                                        #her processer GRU de 20 timesteps med de 64 dimensioner, og opdaterer det næste timesteps "hidden memory"
        h_final = h_final.squeeze(0)                                                        #vi gemmer den sidste hidden, fordi det er svaret. Vi squeezer så vi får [1, 3], som betyder de 3 klasser. Det vil sige den med højest value er prediction

        return self.classify(h_final)            
