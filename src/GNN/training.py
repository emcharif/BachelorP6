import torch
import torch.nn.functional as Function
from torch_geometric.loader import DataLoader
from Classifier import Classifier
from load_datasets import load_datasets

graphs = load_datasets()

dataloader = DataLoader(graphs, batch_size=32, shuffle=True, drop_last=False) #batches graphs together

model = Classifier(input_dim=11, hidden_dim=20, output_dim=3)
opt = torch.optim.Adam(model.parameters()) #updates model weights - leaning rate default = 0.001

for epoch in range(20): #20 passes over dataset
    for batched_graph in dataloader: #iterates over batches of graphs
        logits = model(batched_graph) #raw prediction scores
        loss = Function.cross_entropy(logits, batched_graph.y) #loss function - internally applies softmax
        opt.zero_grad()
        loss.backward() #backpropagation
        opt.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")