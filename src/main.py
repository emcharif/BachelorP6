import sys
import os
import random

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from inject_chain import inject_chain
from load_model import ModelLoader


class Main:

    graphAnalyzer = GraphAnalyzer()
    utilityFunctions = UtilityFunctions()

    load_dotenv()
    key = os.getenv("SECRET_KEY")

    def visualize_watermark(self, dataset_name="PROTEINS"):
 
        rng = random.Random(self.key)

        # ── Load dataset ───────────────────────────────────────────────
        dataset = self.utilityFunctions.load_dataset(name=dataset_name)

        max_length_graphs, graph_index_max = self.graphAnalyzer.get_global_chain_length(dataset)
        print(f"Global chain length for {dataset_name}: {max_length_graphs}")
        min_length_selected_graphs, graph_index_min = self.graphAnalyzer.get_shortest_chain_length(dataset)
        print(f"Minimum chain length for {dataset_name}: {min_length_selected_graphs}")

        is_binary = self.utilityFunctions.is_binary(dataset)
        print(f"Is the dataset {dataset_name} binary? {is_binary}")

        watermarked_graph_max = inject_chain(dataset[graph_index_max], max_length_graphs, is_binary, rng)
        watermarked_graph_min = inject_chain(dataset[graph_index_min], max_length_graphs, is_binary, rng)

        benign_edges_max, delta_edges_max = self.utilityFunctions.dif_watermarked_and_benign_graph_edges(
            selected_graph_edges=dataset[graph_index_max].edge_index.tolist(),
            watermarked_graph_edges=watermarked_graph_max.edge_index.tolist()
        )
        benign_edges_min, delta_edges_min = self.utilityFunctions.dif_watermarked_and_benign_graph_edges(
            selected_graph_edges=dataset[graph_index_min].edge_index.tolist(),
            watermarked_graph_edges=watermarked_graph_min.edge_index.tolist()
        )

        # Fixed: return all four values instead of undefined benign_edges, delta_edges
        return benign_edges_max, delta_edges_max, benign_edges_min, delta_edges_min

    async def check_model(self, model):

        rng = random.Random(self.key)
        model_loader = ModelLoader()

        # Read and load the suspect model once — removed duplicate loading block
        file_bytes = await model.read()
        suspect_model = model_loader.load_model(file_bytes=file_bytes)

        dataset_name = model_loader.identify_dataset(suspect_model)

        dataset = self.utilityFunctions.load_dataset(name=dataset_name)
        global_chain_length, _ = self.graphAnalyzer.get_global_chain_length(dataset)
        is_binary = self.utilityFunctions.is_binary(dataset)

        _, unselected_graphs = self.utilityFunctions.graphs_to_watermark(
            dataset=dataset, rng=rng
        )

        benign_model = model_loader.load_model(f"models/{dataset_name}/benign_model.pth")
        watermarked_model = model_loader.load_model(f"models/{dataset_name}/watermarked_model.pth")

        verification_graphs = []
        for graph in unselected_graphs[:50]:
            modified = inject_chain(graph, global_chain_length, is_binary, rng)
            verification_graphs.append(modified)

        trainer = Trainer(dataset=dataset)

        result = trainer.is_model_trained_on_watermarked_dataset(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            original_dataset=list(dataset),
            watermarked_graphs=verification_graphs
        )

        return result