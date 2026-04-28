import sys
import os
import random

from dotenv import load_dotenv
from torch_geometric.data import Data  # was missing

sys.path.insert(0, os.path.dirname(__file__))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from GNN.Evaluator import Evaluator
from inject_chain import inject_chain
from load_model import ModelLoader


class Main:

    graphAnalyzer = GraphAnalyzer()
    utilityFunctions = UtilityFunctions()

    def visualize_watermark(self, dataset_name="PROTEINS"):
        evaluator = Evaluator()

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
        # Fixed: was discarding unselected_graphs with _
        selected_graphs, unselected_graphs = self.utilityFunctions.graphs_to_watermark(
            dataset=dataset, rng=rng
        )

        watermarked_graphs = []
        for graph in selected_graphs:
            modified_graph = inject_chain(graph, global_chain_length, is_binary, rng)
            watermarked_graphs.append(modified_graph)

        clean_unselected = [
            Data(
                x=g.x,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr if g.edge_attr is not None else None,
                y=g.y
            )
            for g in unselected_graphs
        ]

        complete_dataset = watermarked_graphs + clean_unselected

        # ── Verify watermark structurally ─────────────────────────────
        # Fixed: verify_watermark is on Evaluator, not Trainer
        watermark_present = evaluator.verify_watermark(
            original_dataset=list(dataset),
            watermarked_graphs=watermarked_graphs,
            chain_length=global_chain_length
        )
        print(f"Watermark structurally verified in dataset: {watermark_present}")

        if not watermark_present:
            print("WARNING: Watermark verification failed — chains may not have been injected correctly.")

        # ── Train watermarked model ───────────────────────────────────
        watermarked_trainer = Trainer(dataset=complete_dataset)
        watermarked_model = watermarked_trainer.train(
            enable_prints=True,
            modeltype="watermarked"
        )

        # ── Train benign model ────────────────────────────────────────
        benign_trainer = Trainer(dataset=list(dataset))
        benign_model = benign_trainer.train(
            enable_prints=True,
            modeltype="benign"
        )

        # ── Build verification graphs ─────────────────────────────────
        # Fixed: verification_graphs was used but never built
        verification_graphs = []
        for graph in unselected_graphs[:50]:
            modified = inject_chain(graph, global_chain_length, is_binary, rng)
            verification_graphs.append(modified)

        # ── Check suspect model ───────────────────────────────────────
        suspect_model = watermarked_model

        # Fixed: is_model_trained_on_watermarked_dataset is on Trainer, not benign_trainer
        # and takes original_dataset + watermarked_graphs, not verification_graphs
        behavioural_match = watermarked_trainer.is_model_trained_on_watermarked_dataset(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            original_dataset=list(dataset),
            watermarked_graphs=verification_graphs
        )
        print(f"Suspect model behaviourally matches watermarked model: {behavioural_match}")

        # ── Edge diff ─────────────────────────────────────────────────
        # Fixed: was missing self.
        benign_edges, delta_edges = self.utilityFunctions.dif_watermarked_and_benign_graph_edges(
            selected_graph_edges=selected_graphs[0].edge_index.tolist(),
            watermarked_graph_edges=watermarked_graphs[0].edge_index.tolist()
        )

        return benign_edges, delta_edges

    def check_model(self, model_file):
        """
        Accepts an uploaded .pth file, loads it as a suspect model,
        trains reference models, and returns the behavioural match p-value.
        """
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        dataset_name = "PROTEINS"
        dataset = self.utilityFunctions.load_dataset(name=dataset_name)
        global_chain_length = self.graphAnalyzer.get_global_chain_length(dataset)
        is_binary = self.utilityFunctions.is_binary(dataset)

        selected_graphs, unselected_graphs = self.utilityFunctions.graphs_to_watermark(
            dataset=dataset, rng=rng
        )

        watermarked_graphs = []
        for graph in selected_graphs:
            modified_graph = inject_chain(graph, global_chain_length, is_binary, rng)
            watermarked_graphs.append(modified_graph)

        clean_unselected = [
            Data(x=g.x, edge_index=g.edge_index,
                 edge_attr=g.edge_attr if g.edge_attr is not None else None, y=g.y)
            for g in unselected_graphs
        ]

        complete_dataset = watermarked_graphs + clean_unselected

        watermarked_trainer = Trainer(dataset=complete_dataset)
        watermarked_model = watermarked_trainer.train(enable_prints=False, modeltype="watermarked")

        benign_trainer = Trainer(dataset=list(dataset))
        benign_model = benign_trainer.train(enable_prints=False, modeltype="benign")

        verification_graphs = []
        for graph in unselected_graphs[:50]:
            modified = inject_chain(graph, global_chain_length, is_binary, rng)
            verification_graphs.append(modified)

        file_bytes = model_file.file.read()
        loader = ModelLoader()
        suspect_model = loader.load_model(file_bytes=file_bytes)

        result = watermarked_trainer.is_model_trained_on_watermarked_dataset(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            original_dataset=list(dataset),
            watermarked_graphs=verification_graphs
        )

        return result


if __name__ == "__main__":
    main = Main()
    main.visualize_watermark(dataset_name="PROTEINS")