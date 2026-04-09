import torch
import random
import os

from torch_geometric.loader import DataLoader
from GNN.Classifier import Classifier
from dotenv import load_dotenv
from torch_geometric.data import Batch
from scipy.stats import binomtest

import torch.nn.functional as Function

class Trainer:
    def __init__(self, dataset: list=None, batch_size=64, train_pct=0.7, val_pct=0.15, learning_rate=0.001, hidden_dim=128, epochs=50):

        self.dataset = dataset
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

        self.organize_dataset()

    def organize_dataset(self):

        dataset = self.dataset

        load_dotenv()
        key = os.getenv("SECRET_KEY")
        indices = list(range(len(dataset)))
        rng = random.Random(key)
        rng.shuffle(indices)

        train_size = int(self.train_pct * len(dataset))
        val_size   = int(self.val_pct   * len(dataset))

        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        self.train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader([dataset[i] for i in val_idx], batch_size=self.batch_size)
        self.test_loader  = DataLoader([dataset[i] for i in test_idx], batch_size=self.batch_size)

        self.input_dim = dataset[0].x.shape[1]
        self.output_dim = int(max(graph.y.item() for graph in dataset)) + 1

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

    def train(self, enable_prints: bool = False, modeltype: str = None):
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

        print(f"Final Test Accuracy (modeltype: {modeltype}): {self.evaluate(self.test_loader):.4f}")    
        return self.model
    
    def get_predictions(self, model, dataset: list):
        model.eval()
        predictions = []
        confidences = []
        with torch.no_grad():
            for graph in dataset:
                batch = Batch.from_data_list([graph])
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs.max(dim=1).values.item()
                predictions.append(pred)
                confidences.append(conf)
        return predictions, confidences

    def is_model_trained_on_watermarked_dataset(self, benign_model, watermarked_model, suspect_model, watermarked_graphs: list):

        benign_predictions, benign_confidences = self.get_predictions(benign_model, watermarked_graphs)
        watermarked_predictions, watermarked_confidences = self.get_predictions(watermarked_model, watermarked_graphs)
        suspect_predictions, suspect_confidences = self.get_predictions(suspect_model, watermarked_graphs)

        print(f"benign avg confidence:      {sum(benign_confidences)/len(benign_confidences):.2f}")
        print(f"watermarked avg confidence: {sum(watermarked_confidences)/len(watermarked_confidences):.2f}")
        print(f"suspect avg confidence:     {sum(suspect_confidences)/len(suspect_confidences):.2f}")

        agree_benign = sum(suspect_prediction == benign_prediction for suspect_prediction, benign_prediction in zip(suspect_predictions, benign_predictions)) / len(suspect_predictions)
        agree_watermark = sum(suspect_prediction == watermarked_prediction for suspect_prediction, watermarked_prediction in zip(suspect_predictions, watermarked_predictions))/ len(suspect_predictions)

        result = binomtest(int(agree_watermark * len(suspect_predictions)), len(suspect_predictions), p=agree_benign)

        if result.pvalue < 0.05:
            return True
        else:
            return False