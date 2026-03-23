import torch
import random
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from src.undirected_graphs.GNN.Classifier import Classifier
from dotenv import load_dotenv
import torch.nn.functional as Function


class Trainer:
    def __init__(self, dataset_name, batch_size=32, train_pct=0.80, val_pct=0.10, learning_rate=0.001, hidden_dim=128, epochs=50):

        self.dataset_name  = dataset_name
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

    def load_data(self, dataset=None):

        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        if isinstance(dataset, list):
            rng.shuffle(dataset)

            self.input_dim  = dataset[0].x.shape[1] if dataset[0].x is not None else 0
            self.output_dim = int(max(graph.y.item() for graph in dataset)) + 1 # +1 because of 0 indexing
        else:
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            dataset = dataset[indices]

            self.input_dim  = dataset.num_node_features
            self.output_dim = dataset.num_classes

        train_size = int(self.train_pct * len(dataset))
        val_size   = int(self.val_pct   * len(dataset))

        self.train_loader = DataLoader(dataset[:train_size],                      batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(dataset[train_size:train_size + val_size], batch_size=self.batch_size)
        self.test_loader  = DataLoader(dataset[train_size + val_size:],           batch_size=self.batch_size)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total   = 0
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

    def get_model(self, modeltype):
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
        return predictions, confidences