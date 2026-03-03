import torch
from Classifier import Classifier

graph = torch.load("src/data/training_dataset/data-0004-0000.pt", weights_only=False)

model = Classifier(input_dim=11, hidden_dim=20, output_dim=3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    from torch_geometric.data import Batch
    batched = Batch.from_data_list([graph])
    logits = model(batched)
    predicted_class = logits.argmax(dim=1).item()

print(f"Predicted label: {predicted_class}")