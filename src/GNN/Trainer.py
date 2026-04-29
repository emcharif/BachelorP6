import torch
import random
import os

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from scipy.stats import binomtest
from GNN.Classifier import Classifier
from dotenv import load_dotenv
import torch.nn.functional as Function

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer


class Trainer:

    utility = UtilityFunctions()
    analyzer = GraphAnalyzer()

    def __init__(
        self,
        dataset: list = None,
        train_dataset: list = None,
        val_dataset: list = None,
        test_dataset: list = None,
        dataset_name: str = None,
        batch_size=64,
        train_pct=0.7,
        val_pct=0.15,
        learning_rate=0.001,
        hidden_dim=128,
        epochs=50,
        seed=None,
        use_watermark_head: bool = False,
        watermark_loss_weight: float = 1.0,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.input_dim = None
        self.output_dim = None

        self.use_watermark_head = use_watermark_head
        self.watermark_loss_weight = watermark_loss_weight

        if (
            self.train_dataset is not None
            and self.val_dataset is not None
            and self.test_dataset is not None
        ):
            self.organize_explicit_splits()
        else:
            self.organize_dataset()

    def _set_torch_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _build_loader(self, dataset, shuffle=False, seed_offset=0):
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
        }
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + seed_offset)
            loader_kwargs["generator"] = generator
        return DataLoader(dataset, **loader_kwargs)

    def _set_dimensions_from_datasets(self, datasets):
        non_empty = [ds for ds in datasets if ds is not None and len(ds) > 0]
        if not non_empty:
            raise ValueError("No non-empty datasets were provided to Trainer")
        reference_graph = non_empty[0][0]
        self.input_dim = reference_graph.x.shape[1]
        all_graphs = []
        for ds in non_empty:
            all_graphs.extend(ds)
        self.output_dim = int(max(graph.y.item() for graph in all_graphs)) + 1

    def organize_explicit_splits(self):
        self._set_dimensions_from_datasets(
            [self.train_dataset, self.val_dataset, self.test_dataset]
        )
        self.train_loader = self._build_loader(
            self.train_dataset, shuffle=True, seed_offset=1
        )
        self.val_loader = self._build_loader(
            self.val_dataset, shuffle=False, seed_offset=2
        )
        self.test_loader = self._build_loader(
            self.test_dataset, shuffle=False, seed_offset=3
        )

    def organize_dataset(self):
        dataset = self.dataset
        if dataset is None or len(dataset) == 0:
            raise ValueError("Trainer requires a non-empty dataset")

        load_dotenv()
        key = os.getenv("SECRET_KEY")
        indices = list(range(len(dataset)))

        if self.seed is not None:
            rng = random.Random(self.seed)
        else:
            rng = random.Random(key)

        rng.shuffle(indices)
        train_size = int(self.train_pct * len(dataset))
        val_size = int(self.val_pct * len(dataset))

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        self.train_dataset = [dataset[i] for i in train_idx]
        self.val_dataset = [dataset[i] for i in val_idx]
        self.test_dataset = [dataset[i] for i in test_idx]

        self._set_dimensions_from_datasets(
            [self.train_dataset, self.val_dataset, self.test_dataset]
        )
        self.train_loader = self._build_loader(
            self.train_dataset, shuffle=True, seed_offset=1
        )
        self.val_loader = self._build_loader(
            self.val_dataset, shuffle=False, seed_offset=2
        )
        self.test_loader = self._build_loader(
            self.test_dataset, shuffle=False, seed_offset=3
        )

    def evaluate(self, loader):
        """Evaluate classification accuracy.

        Safe for both plain models and models with use_watermark_head=True,
        which return a (class_logits, wm_score) tuple from forward().
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch)
                # Guard: watermark head active → forward returns a tuple
                if isinstance(out, tuple):
                    out = out[0]
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        return correct / total if total > 0 else 0.0

    def train(self, enable_prints: bool = False, modeltype: str = None):
        self._set_torch_seed()
        self.model = Classifier(self.input_dim, self.hidden_dim, self.output_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in self.train_loader:
                if self.use_watermark_head and hasattr(batch, "is_watermarked"):
                    class_logits, wm_scores = self.model(
                        batch,
                        return_watermark_score=True,
                    )

                    class_loss = Function.cross_entropy(class_logits, batch.y)

                    wm_targets = batch.is_watermarked.view(-1, 1).float()
                    wm_loss = Function.binary_cross_entropy(wm_scores, wm_targets)

                    loss = class_loss + self.watermark_loss_weight * wm_loss
                else:
                    class_logits = self.model(batch)
                    loss = Function.cross_entropy(class_logits, batch.y)

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

        final_acc = self.evaluate(self.test_loader)
        print(f"Final Test Accuracy (modeltype: {modeltype}): {final_acc:.4f}")

        if self.dataset_name is not None and modeltype is not None:
            model_dir = f"models/{self.dataset_name}"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.model.state_dict(), f"{model_dir}/{modeltype}_model.pth")

        return self.model

    def get_predictions(self, model, dataset: list):
        model.eval()
        predictions = []
        confidences = []
        with torch.no_grad():
            for graph in dataset:
                batch = Batch.from_data_list([graph])
                out = model(batch)
                # Guard: watermark head active → forward returns a tuple
                if isinstance(out, tuple):
                    out = out[0]
                probs = torch.softmax(out, dim=1)
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

        analyzer = GraphAnalyzer()

        indices = list(range(len(original_dataset)))
        rng.shuffle(indices)

        benign_preds, benign_confs = self.get_predictions(benign_model, watermarked_graphs)
        watermarked_preds, watermarked_confs = self.get_predictions(watermarked_model, watermarked_graphs)
        suspect_preds, suspect_confs = self.get_predictions(suspect_model, watermarked_graphs)

        print(f"benign avg confidence:      {sum(benign_confs)/len(benign_confs):.2f}")
        print(f"watermarked avg confidence: {sum(watermarked_confs)/len(watermarked_confs):.2f}")
        print(f"suspect avg confidence:     {sum(suspect_confs)/len(suspect_confs):.2f}")

        node_level_agreements = []

        for i, graph in enumerate(watermarked_graphs):
            _, chain_starts, neighbors = analyzer.search_graph(graph)

            if len(chain_starts) != 0:
                dangling = []
                for d in chain_starts:
                    length, edge_node = analyzer.get_dangling_chain_length(d, neighbors)
                    dangling.append((d, length, edge_node))
                max_length = max(dangling, key=lambda x: x[1])
                longest = [d for d in dangling if d[1] == max_length[1]]
            else:
                longest = [(node, 0, node) for node in neighbors.keys()]

            rng.shuffle(longest)

            node_level_agreements.append(suspect_preds[i] == watermarked_preds[i])

        agree_benign = sum(
            s == b for s, b in zip(suspect_preds, benign_preds)
        ) / len(suspect_preds)

        agree_watermark_count = sum(node_level_agreements)

        result = binomtest(agree_watermark_count, len(watermarked_graphs), p=agree_benign)

        print(f"Agreement with watermarked model: {agree_watermark_count}/{len(watermarked_graphs)}")
        print(f"Agreement with benign model (baseline p): {agree_benign:.2f}")
        print(f"p-value: {result.pvalue:.4f}")

        return result.pvalue < 0.05

    def verify_watermark(
        self,
        original_dataset: list,
        watermarked_graphs: list,
        chain_length: int
    ) -> bool:
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        analyzer = GraphAnalyzer()

        indices = list(range(len(original_dataset)))
        rng.shuffle(indices)

        verified = 0

        for i, graph in enumerate(watermarked_graphs):
            _, chain_starts, neighbors = analyzer.search_graph(graph)

            if len(chain_starts) != 0:
                dangling = []
                for d in chain_starts:
                    length, edge_node = analyzer.get_dangling_chain_length(d, neighbors)
                    dangling.append((d, length, edge_node))
                max_length = max(dangling, key=lambda x: x[1])
                longest = [d for d in dangling if d[1] == max_length[1]]
            else:
                longest = [(node, 0, node) for node in neighbors.keys()]

            rng.shuffle(longest)
            selected_node = longest[0]
            expected_edge_node = selected_node[2]

            actual_length, tip = analyzer.get_dangling_chain_length(expected_edge_node, neighbors)

            if actual_length >= chain_length:
                verified += 1

        ratio = verified / len(watermarked_graphs) if watermarked_graphs else 0
        print(f"Watermark verification: {verified}/{len(watermarked_graphs)} graphs confirmed ({ratio:.0%})")
        return ratio > 0.8