import torch
import torch.nn.functional as Function
from TemporalClassifier import TemporalClassifier
from load_datasets import load_datasets

NUM_TIMESTEPS = 20

graphs = load_datasets()

def identity_collate(batch):
    return batch  # prevent PyG from merging internal structure

dataloader = torch.utils.data.DataLoader(
    graphs,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    collate_fn=identity_collate
)

model = TemporalClassifier(input_dim=11, hidden_dim=64, output_dim=3, num_timesteps=NUM_TIMESTEPS)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    total_loss = 0
    for batch in dataloader:
        all_logits = []
        all_labels = []
        for graph in batch:
            logits = model(graph)
            all_logits.append(logits)
            all_labels.append(graph.y)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        loss = Function.cross_entropy(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "model_temporal.pth")