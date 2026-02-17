import torch

data = torch.load(
    "data/graph_dataset/graph_dataset/data-0002-0000.pt",
    weights_only=False
)

print("Type:", type(data))
print("Data:", data)
print()
