import os
import random

from dotenv import load_dotenv

from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.GNN.Evaluator import Evaluator
from src.inject_chain import inject_chain
from src.load_model import ModelLoader
from fastapi import UploadFile, File

class Main:

    TRAIN_PCT = 0.70
    VAL_PCT = 0.15

    graph_analyzer = GraphAnalyzer()
    utility_functions = UtilityFunctions()
    evaluator = Evaluator()

    load_dotenv()
    key = os.getenv("SECRET_KEY")

    def visualize_watermark(self, dataset_name: str) -> tuple[tuple[list, list], tuple[list, list], tuple[list, list], tuple[list, list]]:
        """
        args:
            dataset_name: str, given from frontend user
        return:
            Example of edge input: [[1,2,3,4],[2,1,4,1]] - [[source nodes],[destination nodes]]
            benign_edges_max: edges of the benign graph with the longest dangling chain
            delta_edges_max: edges difference between benign and watermarked versions of that graph
            benign_edges_min: edges of the benign graph with the shortest dangling chain
            delta_edges_min: edges difference between benign and watermarked versions of that graph
        """
        rng = random.Random(self.key)

        dataset = self.utility_functions.load_dataset(dataset_name)

        target_chain_length, graph_index_max = self.graph_analyzer.get_longest_global_chain_length(dataset)
        graph_index_min = self.graph_analyzer.get_shortest_global_chain_length(dataset)

        is_binary = self.utility_functions.is_binary(dataset)
        
        # Longest benign chain and shortest watermarked chain injected
        watermarked_graph_max = inject_chain(dataset[graph_index_max], target_chain_length, is_binary, rng, "subtle")
        
        # Shortest benign chain and longest watermarked chain injected
        watermarked_graph_min = inject_chain(dataset[graph_index_min], target_chain_length, is_binary, rng, "subtle")

        benign_edges_max = dataset[graph_index_max].edge_index.tolist()
        delta_edges_max = self.utility_functions.dif_watermarked_and_benign_graph_edges(
            benign_edges=dataset[graph_index_max].edge_index.tolist(),
            watermarked_edges=watermarked_graph_max.edge_index.tolist()
        )
        benign_edges_min = dataset[graph_index_min].edge_index.tolist()
        delta_edges_min = self.utility_functions.dif_watermarked_and_benign_graph_edges(
            benign_edges=benign_edges_min,
            watermarked_edges=watermarked_graph_min.edge_index.tolist()
        )

        return benign_edges_max, delta_edges_max, benign_edges_min, delta_edges_min

    async def check_model(self, model: UploadFile = File(...)) -> dict:
        """
        Evaluates whether an uploaded model was trained on a watermarked dataset.

        Loads the suspect model from the uploaded file, identifies which dataset
        it was trained on, then constructs unseen watermarked graphs using the secret key. The suspect is then compared
        against a benign and a watermarked reference model using confidence-based
        statistical testing.

        Args:
            model: The uploaded .pth model file to evaluate.

        Returns:
            A dict of evaluation metrics from test_models_with_watermark,
            including average confidences, distances, and paired t-test results
            for the suspect against both the benign and watermarked reference models.
        """

        rng = random.Random(self.key)
        model_loader = ModelLoader()

        file_bytes = await model.read()
        suspect_model = model_loader.load_model(file_bytes=file_bytes)

        dataset_name = model_loader.identify_dataset(suspect_model)

        dataset = self.utility_functions.load_dataset(name=dataset_name)
        global_chain_length, graph_index = self.graph_analyzer.get_longest_global_chain_length(dataset)
        is_binary = self.utility_functions.is_binary(dataset)

        train_clean, _, _ = self.utility_functions.split_dataset(dataset, rng, self.TRAIN_PCT, self.VAL_PCT)

        selected_graphs, _ = self.utility_functions.graphs_to_watermark_same_label(dataset=list(train_clean), graph_index=graph_index, rng=rng)

        verification_graphs = []
        for graph in selected_graphs:
            modified_graph = inject_chain(graph=graph, target_chain_length=global_chain_length, is_binary=is_binary, rng=rng, feature_mode="subtle")
            verification_graphs.append(modified_graph)

        benign_model = model_loader.load_model(f"models/{dataset_name}/benign_model.pth")
        watermarked_model = model_loader.load_model(f"models/{dataset_name}/watermarked_model.pth")

        result = self.evaluator.test_models_with_watermark(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            watermarked_graphs=verification_graphs
        )

        return result