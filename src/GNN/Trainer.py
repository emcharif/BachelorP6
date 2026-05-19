import os
import random
import torch
import torch.nn.functional as Function

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from src.GNN.Classifier import Classifier


class Trainer:
    def __init__(
        self,
        train_dataset: list[Data],
        val_dataset: list[Data],
        test_dataset: list[Data],
        dataset_name: str = None,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        hidden_dim: int = 128,
        epochs: int = 50,
        seed: int = None,
    ):
        """
        Trainer for controlled experiments.

        The dataset must already be split before creating the Trainer.
        This avoids accidentally giving benign and watermarked models different
        validation/test splits.
        """
        if train_dataset is None or len(train_dataset) == 0:
            raise ValueError("Trainer requires a non-empty train_dataset")
        if val_dataset is None:
            raise ValueError("Trainer requires val_dataset")
        if test_dataset is None or len(test_dataset) == 0:
            raise ValueError("Trainer requires a non-empty test_dataset")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed

        self.model = None

        all_graphs = list(train_dataset) + list(val_dataset) + list(test_dataset)

        self.input_dim = train_dataset[0].x.shape[1]
        self.output_dim = int(max(graph.y.item() for graph in all_graphs)) + 1

        self.train_loader = self._build_loader(
            self.train_dataset,
            shuffle=True,
            seed_offset=1,
        )
        self.val_loader = self._build_loader(
            self.val_dataset,
            shuffle=False,
            seed_offset=2,
        )
        self.test_loader = self._build_loader(
            self.test_dataset,
            shuffle=False,
            seed_offset=3,
        )

    def _build_loader(
        self,
        dataset: list[Data],
        shuffle: bool = False,
        seed_offset: int = 0,
    ) -> DataLoader:
        """Builds a DataLoader with optional deterministic shuffling."""
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + seed_offset)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                generator=generator,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train(self, modeltype: str = None) -> Classifier:
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.model = Classifier(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
        )

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        for _ in range(self.epochs):
            self.model.train()

            for batch in self.train_loader:
                class_logits = self.model(batch)
                loss = Function.cross_entropy(class_logits, batch.y)

                opt.zero_grad()
                loss.backward()
                opt.step()

        if self.dataset_name is not None and modeltype is not None:
            model_dir = f"models/{self.dataset_name}"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                self.model.state_dict(),
                f"{model_dir}/{modeltype}_model.pth",
            )

        return self.model