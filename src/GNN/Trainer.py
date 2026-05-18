import torch
import random
import os

from torch_geometric.loader import DataLoader
from src.GNN.Classifier import Classifier
import torch.nn.functional as Function
from torch_geometric.data import Data

class Trainer:

    def __init__(
        self,
        dataset: list = None,
        dataset_name: str = None,
        watermarked_graphs: list = None,
        batch_size=64,
        train_pct=0.7,
        val_pct=0.15,
        learning_rate=0.001,
        hidden_dim=128,
        epochs=50,
        seed=None,
    ):
        """
        Initializes the Trainer, sets up the model dimensions from the dataset,
        and organizes the dataset into train, validation, and test splits.

        Args:
            dataset: The full list of graphs to train on.
            dataset_name: Name of the dataset, used for saving the model to disk.
            watermarked_graphs: Optional list of watermarked graphs to append to
                the training split.
            batch_size: Number of graphs per batch during training.
            train_pct: Fraction of the dataset used for training.
            val_pct: Fraction of the dataset used for validation.
            learning_rate: Step size for the Adam optimizer.
            hidden_dim: Size of the hidden feature vectors in the GNN layers.
            epochs: Number of full passes through the training data.
            seed: Random seed for reproducibility. If None, falls back to SECRET_KEY.
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.watermarked_graphs = watermarked_graphs

        self.batch_size = batch_size
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.input_dim = dataset[0].x.shape[1]
        self.hidden_dim = hidden_dim
        self.output_dim = int(max(graph.y.item() for graph in dataset)) + 1

        self.organize_dataset()

    def _build_loader(self, dataset: list[Data], shuffle=False, seed_offset=0) -> DataLoader:
        """
        Builds a DataLoader for the given dataset.

        If a seed is set, attaches a seeded Generator to ensure reproducible
        batch ordering across runs.

        Args:
            dataset: List of graphs to load.
            shuffle: Whether to shuffle the dataset each epoch.
            seed_offset: Offset added to the seed when creating the Generator,
                ensuring train, val, and test loaders have different random states.

        Returns:
            A DataLoader instance for the given dataset.
        """
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + seed_offset)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def organize_dataset(self) -> None:
        """
        Splits the dataset into train, validation, and test sets and builds
        their corresponding DataLoaders.

        Uses the seed if provided, otherwise falls back to the SECRET_KEY
        environment variable to control the shuffle. Watermarked graphs are
        appended to the training split if provided.

        Raises:
            ValueError: If the dataset is None or empty.
        """
        dataset = self.dataset
        if dataset is None or len(dataset) == 0:
            raise ValueError("Trainer requires a non-empty dataset")

        key = os.getenv("SECRET_KEY")
        indices = list(range(len(dataset)))

        rng = random.Random(self.seed) if self.seed is not None else random.Random(key)
        rng.shuffle(indices)

        train_size = int(self.train_pct * len(dataset))
        val_size = int(self.val_pct * len(dataset))

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        self.train_dataset = [dataset[i] for i in train_idx]
        if self.watermarked_graphs is not None:
            self.train_dataset = self.train_dataset + self.watermarked_graphs
        self.val_dataset = [dataset[i] for i in val_idx]
        self.test_dataset = [dataset[i] for i in test_idx]

        self.train_loader = self._build_loader(self.train_dataset, shuffle=True, seed_offset=1)
        self.val_loader = self._build_loader(self.val_dataset, shuffle=False, seed_offset=2)
        self.test_loader = self._build_loader(self.test_dataset, shuffle=False, seed_offset=3)

    def train(self, modeltype: str = None) -> Classifier:
        """
        Trains the GNN classifier on the training split.

        Sets random seeds for reproducibility, initializes the Classifier model
        and Adam optimizer, then runs the training loop for the configured number
        of epochs.

        Saves the trained model to disk under models/{dataset_name}/{modeltype}_model.pth
        if both dataset_name and modeltype are provided.

        Args:
            modeltype: Label used for the saved model filename, e.g. 'benign'
                or 'watermarked'. If None, the model is not saved to disk.

        Returns:
            The trained Classifier model in its final state.
        """
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.model = Classifier(self.input_dim, self.hidden_dim, self.output_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
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
            torch.save(self.model.state_dict(), f"{model_dir}/{modeltype}_model.pth")

        return self.model