import torch
import random
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from GNN.Classifier import Classifier
from dotenv import load_dotenv
import torch.nn.functional as Function


class Trainer:
    def __init__(self, dataset: list, batch_size=32, train_pct=0.70, val_pct=0.15, learning_rate=0.001, hidden_dim=64, epochs=30):

        self.dataset       = dataset
        self.batch_size    = batch_size
        self.train_pct     = train_pct
        self.val_pct       = val_pct
        self.learning_rate = learning_rate
        self.hidden_dim    = hidden_dim
        self.epochs        = epochs

        self.model         = None
        self.train_loader  = None
        self.val_loader    = None
        self.test_loader   = None

        self.organize_dataset()

    def organize_dataset(self):

        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        rng.shuffle(self.dataset)

        self.input_dim  = self.dataset[0].vehicle.x.shape[1]
        #self.output_dim = int(max(graph.y.item() for graph in self.dataset)) + 1 # +1 because of 0 indexing
        self.output_dim = 3 #temporary

        train_size = int(self.train_pct * len(self.dataset))
        val_size   = int(self.val_pct   * len(self.dataset))

        self.train_loader = torch.utils.data.DataLoader(self.dataset[:train_size], batch_size=self.batch_size, shuffle=True, drop_last=False, collate_fn=identity_collate)
        self.val_loader = torch.utils.data.DataLoader(self.dataset[train_size:train_size + val_size], batch_size=self.batch_size, drop_last=False, collate_fn=identity_collate)
        self.test_loader = torch.utils.data.DataLoader(self.dataset[train_size + val_size:], batch_size=self.batch_size, drop_last=False, collate_fn=identity_collate)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch in loader:
                all_logits = []
                all_labels = []
                for graph in batch:
                    all_logits.append(self.model(graph))
                    all_labels.append(graph.y)
                pred = torch.cat(all_logits, dim=0).argmax(dim=1)
                labels = torch.stack(all_labels).squeeze(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def train(self, enable_prints = False):
        num_timesteps = len(self.dataset[0].vehicle.ptr) - 1

        self.model = Classifier(self.input_dim, self.hidden_dim, self.output_dim, num_timesteps=num_timesteps)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                all_logits = []
                all_labels = []
                for graph in batch:                  
                    logits = self.model(graph)
                    all_logits.append(logits)
                    all_labels.append(graph.y)

                logits = torch.cat(all_logits, dim=0)
                labels = torch.stack(all_labels).squeeze(1)

                loss = Function.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            if enable_prints:
                print(
                    f"Epoch {epoch:02d} | "
                    f"Loss: {total_loss / len(self.train_loader):.4f} | "
                    f"Train Acc: {self.evaluate(self.train_loader):.4f} | "
                    f"Val Acc: {self.evaluate(self.val_loader):.4f}"
                )

    def get_model(self, modeltype=None):
        print(f"Final Test Accuracy {modeltype}: {self.evaluate(self.test_loader):.4f}")
        #torch.save(self.model.state_dict(), f"{self.dataset_name}{modeltype}_model.pth")

        return self.model

    def get_predictions(self, model, graphs):
        model.eval()
        predictions = []
        confidences = []
        with torch.no_grad():
            for graph in graphs:
                batch = Batch.from_data_list([graph])
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs.max(dim=1).values.item()
                predictions.append(pred)
                confidences.append(conf)
        return predictions, 

def identity_collate(batch):                                                                       #normalt merger PyGs Dataloader graferne sammen, her siger vi altså bare at den skal returnere listen som den er 
    return batch 