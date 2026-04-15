import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy

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
    
    def pruning_attack(self, model, pruning_rate=0.2):
        pruned_model = copy.deepcopy(model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_rate)
                prune.remove(module, 'weight')  # make pruning permanent
    
        return pruned_model