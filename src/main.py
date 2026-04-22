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

    def visualize_watermark(self, dataset_name = "ENZYMES"):

        # Seed once — shared across the entire pipeline
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        # ── Load dataset ───────────────────────────────────────────────
        dataset = self.utilityFunctions.load_dataset(name=dataset_name)

        global_chain_length = self.graphAnalyzer.get_global_chain_length(dataset)
        print(f"Global chain length for {dataset_name}: {global_chain_length}")

        is_binary = self.utilityFunctions.is_binary(dataset)
        print(f"Is the dataset {dataset_name} binary? {is_binary}")

        # ── Select and watermark graphs ────────────────────────────────
        selected_graphs, _ = self.utilityFunctions.graphs_to_watermark(
            dataset=dataset, rng=rng
        )

        watermarked_graphs = []
        for graph in selected_graphs:
            modified_graph = inject_chain(graph, global_chain_length, is_binary, rng)
            watermarked_graphs.append(modified_graph)


        # ── Edge diff (first graph as example) ────────────────────────
        benign_edges, delta_edges = self.utilityFunctions.dif_watermarked_and_benign_graph_edges(
            selected_graph_edges=selected_graphs[0].edge_index.tolist(),
            watermarked_graph_edges=watermarked_graphs[0].edge_index.tolist()
        )

        return benign_edges, delta_edges
    
    async def check_model(self, model):

        model_loader = ModelLoader()

        file_bytes = await model.read()
        suspect_model = model_loader.load_model(file_bytes=file_bytes)

        dataset_name = model_loader.identify_dataset(suspect_model)

        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        dataset = self.utilityFunctions.load_dataset(name=dataset_name)
        global_chain_length = self.graphAnalyzer.get_global_chain_length(dataset)
        is_binary = self.utilityFunctions.is_binary(dataset)

        selected_graphs, _ = self.utilityFunctions.graphs_to_watermark(dataset=dataset, rng=rng)

        watermarked_graphs = [
            inject_chain(g, global_chain_length, is_binary, rng)
            for g in selected_graphs
        ]

        # ── Train reference benign + watermarked models ───────────────────────
        benign_model = model_loader.load_model(f"models/{dataset_name}/benign_model.pth")
        watermarked_model = model_loader.load_model(f"models/{dataset_name}/watermarked_model.pth")

        benign_trainer = Trainer(dataset=list(dataset), dataset_name=dataset_name)

        # ── Run behavioural test and retrieve p-value ─────────────────────────
        p_value = benign_trainer.is_model_trained_on_watermarked_dataset(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            original_dataset=list(dataset),
            watermarked_graphs=watermarked_graphs
        )

        return p_value


if __name__ == "__main__":
    main = Main()
    main.main(dataset_name="ENZYMES")