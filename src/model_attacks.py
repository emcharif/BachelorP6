import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

class model_attacks:
    def fine_tune_attack(self, model, attacker_dataset, enable_prints=False, epochs=10, learning_rate=0.001):
        loader = DataLoader(attacker_dataset, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        for epoch in range(epochs):
            total_loss = 0
            model.train()
            for batch in loader:
                logits = model(batch)
                loss = F.cross_entropy(logits, batch.y)
                total_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            if enable_prints:
                print(f"Epoch {epoch:02d} | Loss: {total_loss / len(loader):.4f}")

        return model