import sys
import os
import random
import torch

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from GNN.Evaluator import Evaluator
from torch_geometric.data import Data
from inject_chain import inject_chain


# ── Configuration ──────────────────────────────────────────────────────────────
VERIFICATION_COUNT = 50
WATERMARK_LOSS_WEIGHT = 4.0
# ───────────────────────────────────────────────────────────────────────────────


class Main:
    def main(self, dataset_name="PROTEINS"):

        graphAnalyzer = GraphAnalyzer()
        utilityFunctions = UtilityFunctions()
        evaluator = Evaluator()

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

        # ── 3. Tag graphs before building complete_dataset ────────────────
        # is_watermarked drives both the weighted classification loss and
        # the watermark head's BCE loss in the Trainer.
        for g in watermarked_graphs:
            g.is_watermarked = torch.tensor([1.0])
        for g in clean_unselected:
            g.is_watermarked = torch.tensor([0.0])

        complete_dataset = watermarked_graphs + clean_unselected

        # ── 4. Verify watermark structurally present before training ──────
        watermark_present = evaluator.verify_watermark(
            original_dataset=list(dataset),
            watermarked_graphs=watermarked_graphs,
            chain_length=global_chain_length
        )
        print(f"Watermark structurally verified in dataset: {watermark_present}")

        if not watermark_present:
            print("WARNING: Watermark verification failed.")

        # ── 5. Train watermarked model ────────────────────────────────────
        # modeltype="watermarked" activates the watermark head loss in Trainer
        watermarked_trainer = Trainer(
            dataset=complete_dataset,
            watermark_loss_weight=WATERMARK_LOSS_WEIGHT,
        )
        watermarked_model = watermarked_trainer.train(
            enable_prints=True,
            modeltype="watermarked"
        )

        # ── 6. Train benign model ─────────────────────────────────────────
        # modeltype="benign" — watermark head loss is NOT applied, so the
        # benign model's watermark head stays untrained (outputs ~0.5 noise)
        benign_trainer = Trainer(dataset=list(dataset))
        benign_model = benign_trainer.train(
            enable_prints=True,
            modeltype="benign"
        )

        # ── 7. Build verification graphs ──────────────────────────────────
        n_verification = min(VERIFICATION_COUNT, len(unselected_graphs))
        if n_verification < VERIFICATION_COUNT:
            print(
                f"WARNING: only {n_verification} unselected graphs available "
                f"for verification (requested {VERIFICATION_COUNT})."
            )

        verification_graphs = []
        for graph in unselected_graphs[:n_verification]:
            modified = inject_chain(graph, global_chain_length, is_binary, rng)
            verification_graphs.append(modified)

        print(f"Verification graphs built: {len(verification_graphs)}")

        # ── 8. Test suspect model ─────────────────────────────────────────
        suspect_model = watermarked_model

        test_results = evaluator.test_models_with_watermark(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=suspect_model,
            watermarked_graphs=verification_graphs
        )

        # Detection uses the watermark head scores, not classification confidence.
        # See Evaluator.test_models_with_watermark for full logic.
        behavioural_match = test_results["detected"]
        print(f"Suspect model behaviourally matches watermarked model: {behavioural_match}")

        # ── 9. Edge diff (first graph as example) ─────────────────────────
        benign_edges, watermarked_edges, delta_edges = utilityFunctions.dif_watermarked_and_benign_graph_edges(
            selected_graph_edges=selected_graphs[0].edge_index.tolist(),
            watermarked_graph_edges=watermarked_graphs[0].edge_index.tolist()
        )

        return watermark_present, behavioural_match, benign_edges, watermarked_edges, delta_edges


if __name__ == "__main__":
    main = Main()
    main.main(dataset_name="PROTEINS")