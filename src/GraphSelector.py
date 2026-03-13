import os
import random

from torch_geometric.datasets import TUDataset

class GraphSelector:
    def __init__(self, secret_key: str, percentage: float = 0.05):
        """
        Args:
            secret_key: Secret key for deterministic file selection
            percentage: Fraction of files to watermark (default 5%)
        """
        self.secret_key = secret_key
        self.percentage = percentage
        self._rng = random.Random(secret_key)

    def get_graphs(self, dataset_name: str):
        dataset = TUDataset(root="./data", name=dataset_name)

        indices = list(range(len(dataset)))
        self._rng.shuffle(indices)

        number_of_graphs = max(1, int(len(indices) * self.percentage))
        selected_indices = indices[:number_of_graphs]

        selected_graphs = []
        for i in range(len(selected_indices)):
            selected_graphs.append(dataset[i])

        return selected_graphs

selector = GraphSelector(secret_key=os.getenv("SECRET_KEY"), percentage=0.05)

graphs_to_watermark = selector.get_graphs("REDDIT-BINARY")
print(graphs_to_watermark)