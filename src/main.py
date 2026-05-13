import os
import random

from dotenv import load_dotenv

from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.GNN.Evaluator import Evaluator
from src.inject_chain import inject_chain
from src.load_model import ModelLoader

class Main:

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

    async def check_model(self, model):

        rng = random.Random(self.key)
        model_loader = ModelLoader()

        file_bytes = await model.read()
        suspect_model = model_loader.load_model(file_bytes=file_bytes)

        dataset_name = model_loader.identify_dataset(suspect_model)

        dataset = self.utility_functions.load_dataset(name=dataset_name)
        global_chain_length, _ = self.graph_analyzer.get_longest_global_chain_length(dataset)
        is_binary = self.utility_functions.is_binary(dataset)

        _, unselected_graphs = self.utility_functions.graphs_to_watermark(dataset=dataset, rng=rng)

        verification_graphs = []
        for graph in unselected_graphs[:50]:
            modified = inject_chain(graph, global_chain_length, is_binary, rng, "subtle")
            verification_graphs.append(modified)

        benign_model = model_loader.load_model(f"models/{dataset_name}/benign_model.pth")
        watermarked_model = model_loader.load_model(f"models/{dataset_name}/watermarked_model.pth")

        result = self.evaluator.test_models_with_watermark(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            watermarked_graphs=verification_graphs
        )

        return result