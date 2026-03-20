import torch
import random
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from src.undirected_graphs.GNN.Classifier import Classifier
import torch.nn.functional as Function

class Trainer:
    def __init__(self, dataset_name="MUTAG", batch_size=10, train_pct=0.7, val_pct=0.15, learning_rate=0.001, hidden_dim=128, epochs=20):

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.epochs = epochs

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self, dataset=None, input_dim=None, output_dim=None):
        if dataset is None:
            dataset = TUDataset(root="./data", name=self.dataset_name)

        if isinstance(dataset, list):
            random.shuffle(dataset)
        else:
            dataset = dataset.shuffle()

        train_size = int(self.train_pct * len(dataset))
        val_size = int(self.val_pct * len(dataset))

        self.train_loader = DataLoader(dataset[:train_size], batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(dataset[train_size:train_size + val_size], batch_size=self.batch_size)
        self.test_loader  = DataLoader(dataset[train_size + val_size:], batch_size=self.batch_size)

        self.input_dim  = input_dim  if input_dim  is not None else dataset.num_node_features
        self.output_dim = output_dim if output_dim is not None else dataset.num_classes

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                pred = self.model(batch).argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        return correct / total

    def train(self, enable_prints = False):
        self.model = Classifier(self.input_dim, self.hidden_dim, self.output_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                logits = self.model(batch)
                loss = Function.cross_entropy(logits, batch.y)
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

    def test_and_save(self):
        print(f"Final Test Accuracy: {self.evaluate(self.test_loader):.4f}")
        torch.save(self.model.state_dict(), f"{self.dataset_name}_model.pth")