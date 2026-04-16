from scipy.stats import ttest_rel
import math
import torch
from torch_geometric.data import Batch
from dotenv import load_dotenv
import random
from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
import os

class Evaluator:

    def get_predictions(self, model, dataset: list):
        model.eval()
        predictions = []
        confidences = []
        with torch.no_grad():
            for graph in dataset:
                batch = Batch.from_data_list([graph])
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs.max(dim=1).values.item()
                predictions.append(pred)
                confidences.append(conf)
        return predictions, confidences
    
    def test_models_with_watermark(
        self,
        benign_model,
        watermarked_model,
        suspect_model,
        watermarked_graphs
    ):
        
        if not watermarked_graphs:
            raise ValueError("watermarked_graphs must contain at least one graph")

        _, benign_confs = self.get_predictions(benign_model, watermarked_graphs)
        _, watermarked_confs = self.get_predictions(watermarked_model, watermarked_graphs)
        _, suspect_confs = self.get_predictions(suspect_model, watermarked_graphs)

        benign_avg = sum(benign_confs) / len(benign_confs)
        watermarked_avg = sum(watermarked_confs) / len(watermarked_confs)
        suspect_avg = sum(suspect_confs) / len(suspect_confs)

        total_dist_to_benign = 0.0
        total_dist_to_watermarked = 0.0     

        for suspect, benign, watermarked in zip(suspect_confs, benign_confs, watermarked_confs):
            total_dist_to_benign += abs(suspect - benign)
            total_dist_to_watermarked += abs(suspect - watermarked)

        avg_dist_to_benign = total_dist_to_benign / len(suspect_confs)
        avg_dist_to_watermarked = total_dist_to_watermarked / len(suspect_confs)

        t_stat_vs_benign, p_value_vs_benign = ttest_rel(suspect_confs, benign_confs)
        t_stat_vs_watermarked, p_value_vs_watermarked = ttest_rel(suspect_confs, watermarked_confs)

        results = {
            "benign_avg_confidence": benign_avg,
            "watermarked_avg_confidence": watermarked_avg,
            "suspect_avg_confidence": suspect_avg,
            "avg_distance_to_benign": avg_dist_to_benign,
            "avg_distance_to_watermarked": avg_dist_to_watermarked,
            "t_stat_vs_benign": 0.0 if math.isnan(t_stat_vs_benign) else t_stat_vs_benign,
            "p_value_vs_benign": 1.0 if math.isnan(p_value_vs_benign) else p_value_vs_benign,
            "t_stat_vs_watermarked": 0.0 if math.isnan(t_stat_vs_watermarked) else t_stat_vs_watermarked,
            "p_value_vs_watermarked": 1.0 if math.isnan(p_value_vs_watermarked) else p_value_vs_watermarked,
            "benign_confidences": benign_confs,
            "watermarked_confidences": watermarked_confs,
            "suspect_confidences": suspect_confs,
        }

        print(f"benign avg confidence:           {benign_avg:.4f}")
        print(f"watermarked avg confidence:      {watermarked_avg:.4f}")
        print(f"suspect avg confidence:          {suspect_avg:.4f}")
        print(f"avg distance to benign:          {avg_dist_to_benign:.4f}")
        print(f"avg distance to watermarked:     {avg_dist_to_watermarked:.4f}")
        print(f"paired p-value vs benign:        {results['p_value_vs_benign']:.4f}")
        print(f"paired p-value vs watermarked:   {results['p_value_vs_watermarked']:.4f}")

        return results
    
    def verify_watermark(self, original_dataset: list, watermarked_graphs: list, chain_length: int) -> bool:
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        utility = UtilityFunctions()
        analyzer = GraphAnalyzer()

        # Mirror graphs_to_watermark exactly — same rng, same indices, same selected graphs
        indices = list(range(len(original_dataset)))
        rng.shuffle(indices)

        # watermarked_graphs is already in the same order as selected_idx
        # so we zip them directly
        verified = 0

        for i, graph in enumerate(watermarked_graphs):
            _, chain_starts, neighbors = analyzer.search_graph(graph)

            if len(chain_starts) != 0:
                dangling = []
                for d in chain_starts:
                    length, edge_node = utility.get_dangling_chain_length(d, neighbors)
                    dangling.append((d, length, edge_node))
                max_length = max(dangling, key=lambda x: x[1])
                longest = [d for d in dangling if d[1] == max_length[1]]
            else:
                longest = [(node, 0, node) for node in neighbors.keys()]

            # Mirror inject_chain's node selection — same rng advancing across graphs
            rng.shuffle(longest)
            selected_node = longest[0]
            expected_edge_node = selected_node[2]

            # The injected chain tip is the last node — highest node id in the graph
            # Walk forward from expected_edge_node and verify chain length
            actual_length, tip = utility.get_dangling_chain_length(expected_edge_node, neighbors)

            if actual_length >= chain_length:
                verified += 1

        ratio = verified / len(watermarked_graphs) if watermarked_graphs else 0
        print(f"Watermark verification: {verified}/{len(watermarked_graphs)} graphs confirmed ({ratio:.0%})")
        return ratio > 0.8