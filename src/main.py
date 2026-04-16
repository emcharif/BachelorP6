import sys
import os
import random

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from torch_geometric.data import Data
from inject_chain import inject_chain


class Main:
    def main(self, dataset_name: str):

        graphAnalyzer = GraphAnalyzer()
        utilityFunctions = UtilityFunctions()

        # Seed once — shared across the entire pipeline
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        # ── 1. Load dataset ───────────────────────────────────────────────
        dataset = utilityFunctions.load_dataset(name=dataset_name)

        global_chain_length = graphAnalyzer.get_global_chain_length(dataset)
        print(f"Global chain length for {dataset_name}: {global_chain_length}")

        is_binary = utilityFunctions.is_binary(dataset)
        print(f"Is the dataset {dataset_name} binary? {is_binary}")

        # ── 2. Select and watermark graphs ────────────────────────────────
        selected_graphs, unselected_graphs = utilityFunctions.graphs_to_watermark(
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

        # ── 3. Verify watermark is structurally present before training ───
        # We need a temporary Trainer instance just to access verify_watermark.
        # We pass complete_dataset here and also use it for real training below.
        watermarked_trainer = Trainer(dataset=complete_dataset)

        watermark_present = watermarked_trainer.verify_watermark(
            original_dataset=list(dataset),
            watermarked_graphs=watermarked_graphs,
            chain_length=global_chain_length
        )
        print(f"Watermark structurally verified in dataset: {watermark_present}")

        if not watermark_present:
            print("WARNING: Watermark verification failed — chains may not have been injected correctly.")

        # ── 4. Train watermarked model ────────────────────────────────────
        watermarked_model = watermarked_trainer.train(
            enable_prints=True,
            modeltype="watermarked"
        )

        # ── 5. Train benign model (on original unmodified dataset) ────────
        benign_trainer = Trainer(dataset=list(dataset))
        benign_model = benign_trainer.train(
            enable_prints=True,
            modeltype="benign"
        )

        # ── 6. Check suspect model behaviourally ─────────────────────────
        # Replace suspect_model with the actual suspect model you want to test
        suspect_model = watermarked_model

        behavioural_match = benign_trainer.is_model_trained_on_watermarked_dataset(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            original_dataset=list(dataset),
            watermarked_graphs=watermarked_graphs
        )
        print(f"Suspect model behaviourally matches watermarked model: {behavioural_match}")

        # ── 7. Edge diff (first graph as example) ────────────────────────
        benign_edges, watermarked_edges, delta_edges = utilityFunctions.dif_watermarked_and_benign_graph_edges(
            selected_graph_edges=selected_graphs[0].edge_index.tolist(),
            watermarked_graph_edges=watermarked_graphs[0].edge_index.tolist()
        )

        return watermark_present, behavioural_match, benign_edges, watermarked_edges, delta_edges


if __name__ == "__main__":
    main = Main()
    main.main(dataset_name="ENZYMES")