import torch
import random
import os

from torch_geometric.loader import DataLoader
from GNN.Classifier import Classifier
from dotenv import load_dotenv
import torch.nn.functional as Function


class Trainer:
    def __init__(
        self,
        dataset: list = None,
        train_dataset: list = None,
        val_dataset: list = None,
        test_dataset: list = None,
        batch_size=64,
        train_pct=0.7,
        val_pct=0.15,
        learning_rate=0.001,
        hidden_dim=128,
        epochs=50,
        seed=None,
        watermarked_graphs: list = None,   # kept for API compatibility, unused
        watermark_loss_weight: float = 1.0,
    ):
        self.dataset = dataset
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
        self.watermark_loss_weight = watermark_loss_weight

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.input_dim = None
        self.output_dim = None

        if (
            self.train_dataset is not None
            and self.val_dataset is not None
            and self.test_dataset is not None
        ):
            self.organize_explicit_splits()
        else:
            self.organize_dataset()

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _set_torch_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _build_loader(self, dataset, shuffle=False, seed_offset=0):
        loader_kwargs = {"batch_size": self.batch_size, "shuffle": shuffle}
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

        self.train_dataset = [dataset[i] for i in indices[:train_size]]
        self.val_dataset   = [dataset[i] for i in indices[train_size:train_size + val_size]]
        self.test_dataset  = [dataset[i] for i in indices[train_size + val_size:]]

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

    # ──────────────────────────────────────────────────────────────────
    # Evaluation — uses only the classification head
    # ──────────────────────────────────────────────────────────────────

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                # Classifier now returns (class_logits, watermark_score)
                class_logits, _ = self.model(batch)
                pred = class_logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        return correct / total if total > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self, enable_prints: bool = False, modeltype: str = None):
        self._set_torch_seed()
        self.model = Classifier(self.input_dim, self.hidden_dim, self.output_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        use_weighted_loss = self.watermark_loss_weight != 1.0
        has_watermark_head = modeltype == "watermarked"

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in self.train_loader:
                class_logits, wm_scores = self.model(batch)

                # ── Classification loss (always) ──────────────────────────
                if use_weighted_loss and hasattr(batch, "is_watermarked"):
                    per_sample_loss = Function.cross_entropy(
                        class_logits, batch.y, reduction="none"
                    )
                    wm_flags = batch.is_watermarked.view(-1).float()
                    per_graph_weight = 1.0 + (self.watermark_loss_weight - 1.0) * wm_flags
                    per_graph_weight = per_graph_weight / per_graph_weight.mean()
                    class_loss = (per_sample_loss * per_graph_weight).mean()
                else:
                    class_loss = Function.cross_entropy(class_logits, batch.y)

                # ── Watermark head loss (watermarked model only) ───────────
                # Binary cross-entropy: push wm_score → 1.0 for watermarked
                # graphs, → 0.0 for clean graphs.
                # Only trains when is_watermarked tags are present (i.e. when
                # this is the watermarked model — benign trainer never tags).
                if has_watermark_head and hasattr(batch, "is_watermarked"):
                    wm_targets = batch.is_watermarked.view(-1, 1).float()
                    wm_loss = Function.binary_cross_entropy(
                        wm_scores, wm_targets
                    )
                    # Scale: watermark loss weighted same as classification.
                    # This keeps the two losses balanced — wm_head doesn't
                    # dominate or get ignored.
                    loss = class_loss + self.watermark_loss_weight * wm_loss
                else:
                    loss = class_loss

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

        print(
            f"Final Test Accuracy (modeltype: {modeltype}): "
            f"{self.evaluate(self.test_loader):.4f}"
        )
        return self.model