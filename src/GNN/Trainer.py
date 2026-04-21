import torch
import random
import os

from torch_geometric.loader import DataLoader
from GNN.Classifier import Classifier
from dotenv import load_dotenv
from torch_geometric.data import Batch
from scipy.stats import binomtest

import torch.nn.functional as Function
from graph_analyzer import GraphAnalyzer
from utils import UtilityFunctions

class Trainer:

    utility = UtilityFunctions()
    analyzer = GraphAnalyzer()

    def __init__(self, dataset: list=None, dataset_name: str=None, batch_size=64, train_pct=0.7, val_pct=0.15, learning_rate=0.001, hidden_dim=128, epochs=50):

        self.dataset = dataset
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
        torch.save(self.model.state_dict(), f"models/{self.dataset_name}/{modeltype}_model.pth")  
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

    def is_model_trained_on_watermarked_dataset(
    self,
    benign_model,
    watermarked_model,
    suspect_model,
    original_dataset: list,
    watermarked_graphs: list
    ):
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        # Step 1: Re-derive selected graph indices from the key
        indices = list(range(len(original_dataset)))
        rng.shuffle(indices)

        # Step 2: Get predictions from all three models on the watermarked graphs
        benign_preds, benign_confs         = self.get_predictions(benign_model, watermarked_graphs)
        watermarked_preds, watermarked_confs = self.get_predictions(watermarked_model, watermarked_graphs)
        suspect_preds, suspect_confs       = self.get_predictions(suspect_model, watermarked_graphs)

        print(f"benign avg confidence:      {sum(benign_confs)/len(benign_confs):.2f}")
        print(f"watermarked avg confidence: {sum(watermarked_confs)/len(watermarked_confs):.2f}")
        print(f"suspect avg confidence:     {sum(suspect_confs)/len(suspect_confs):.2f}")

        # Step 3: Per-graph, re-derive which node was injected and check suspect vs watermarked agreement
        node_level_agreements = []

        for i, graph in enumerate(watermarked_graphs):
            _, chain_starts, neighbors = self.analyzer.search_graph(graph)

            if len(chain_starts) != 0:
                dangling = []
                for d in chain_starts:
                    length, edge_node = self.analyzer.get_dangling_chain_length(d, neighbors)
                    dangling.append((d, length, edge_node))
                max_length = max(dangling, key=lambda x: x[1])
                longest = [d for d in dangling if d[1] == max_length[1]]
            else:
                longest = [(node, 0, node) for node in neighbors.keys()]

            rng.shuffle(longest)  # advances rng — mirrors inject_chain

            # Did suspect agree with watermarked model on this key-selected graph?
            node_level_agreements.append(suspect_preds[i] == watermarked_preds[i])

        # Step 4: Binomial test — is suspect's agreement with watermarked model
        # significantly higher than its agreement with the benign model?
        agree_benign = sum(
            s == b for s, b in zip(suspect_preds, benign_preds)
        ) / len(suspect_preds)

        agree_watermark_count = sum(node_level_agreements)

        result = binomtest(agree_watermark_count, len(watermarked_graphs), p=agree_benign)

        print(f"Agreement with watermarked model: {agree_watermark_count}/{len(watermarked_graphs)}")
        print(f"Agreement with benign model (baseline p): {agree_benign:.2f}")
        print(f"p-value: {result.pvalue:.4f}")

        return result.pvalue < 0.05
        
    def verify_watermark(self, original_dataset: list, watermarked_graphs: list, chain_length: int) -> bool:
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        # Mirror graphs_to_watermark exactly — same rng, same indices, same selected graphs
        indices = list(range(len(original_dataset)))
        rng.shuffle(indices)

        # watermarked_graphs is already in the same order as selected_idx
        # so we zip them directly
        verified = 0

        for i, graph in enumerate(watermarked_graphs):
            _, chain_starts, neighbors = self.analyzer.search_graph(graph)

            if len(chain_starts) != 0:
                dangling = []
                for d in chain_starts:
                    length, edge_node = self.analyzer.get_dangling_chain_length(d, neighbors)
                    dangling.append((d, length, edge_node))
                max_length = max(dangling, key=lambda x: x[1])
                longest = [d for d in dangling if d[1] == max_length[1]]
            else:
                longest = [(node, 0, node) for node in neighbors.keys()]

            # Mirror inject_chain's node selection — same rng advancing across graphs
            rng.shuffle(longest)
            selected_node = longest[0]
            expected_edge_node = selected_node[2]

            # The injected chain tip is the last node — highest node id in the graph
            # Walk forward from expected_edge_node and verify chain length
            actual_length, tip = self.analyzer.get_dangling_chain_length(expected_edge_node, neighbors)

            if actual_length >= chain_length:
                verified += 1

        ratio = verified / len(watermarked_graphs) if watermarked_graphs else 0
        print(f"Watermark verification: {verified}/{len(watermarked_graphs)} graphs confirmed ({ratio:.0%})")
        return ratio > 0.8