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
        """
        Returns (predictions, confidences, watermark_scores) for each graph.

        - predictions:      argmax class index
        - confidences:      max softmax probability (classification confidence)
        - watermark_scores: watermark head output in [0, 1]
                            1.0 = model recognises watermark pattern
                            0.0 = model does not recognise it
        """
        model.eval()
        predictions = []
        confidences = []
        watermark_scores = []

        with torch.no_grad():
            for graph in dataset:
                batch = Batch.from_data_list([graph])
                class_logits, wm_score = model(batch)

                probs = torch.softmax(class_logits, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs.max(dim=1).values.item()

                predictions.append(pred)
                confidences.append(conf)
                watermark_scores.append(wm_score.item())

        return predictions, confidences, watermark_scores

    def test_models_with_watermark(
        self,
        benign_model,
        watermarked_model,
        suspect_model,
        watermarked_graphs
    ):
        if not watermarked_graphs:
            raise ValueError("watermarked_graphs must contain at least one graph")

        _, benign_confs, benign_wm_scores = self.get_predictions(
            benign_model, watermarked_graphs
        )
        _, watermarked_confs, watermarked_wm_scores = self.get_predictions(
            watermarked_model, watermarked_graphs
        )
        _, suspect_confs, suspect_wm_scores = self.get_predictions(
            suspect_model, watermarked_graphs
        )

        # ── Confidence stats (kept for reference) ─────────────────────────
        benign_avg_conf = sum(benign_confs) / len(benign_confs)
        watermarked_avg_conf = sum(watermarked_confs) / len(watermarked_confs)
        suspect_avg_conf = sum(suspect_confs) / len(suspect_confs)

        # ── Watermark head stats (primary detection signal) ───────────────
        # The watermarked model's head is trained to output ~1.0 on
        # watermarked graphs. The benign model's head is untrained so it
        # outputs noise near 0.5. A suspect trained on watermarked data
        # will output scores close to the watermarked model.
        benign_avg_wm = sum(benign_wm_scores) / len(benign_wm_scores)
        watermarked_avg_wm = sum(watermarked_wm_scores) / len(watermarked_wm_scores)
        suspect_avg_wm = sum(suspect_wm_scores) / len(suspect_wm_scores)

        # Distance metrics on watermark scores
        total_dist_to_benign_wm = sum(
            abs(s - b) for s, b in zip(suspect_wm_scores, benign_wm_scores)
        )
        total_dist_to_watermarked_wm = sum(
            abs(s - w) for s, w in zip(suspect_wm_scores, watermarked_wm_scores)
        )
        avg_dist_to_benign_wm = total_dist_to_benign_wm / len(suspect_wm_scores)
        avg_dist_to_watermarked_wm = total_dist_to_watermarked_wm / len(suspect_wm_scores)

        # Paired t-tests on watermark head scores
        t_vs_benign, p_vs_benign = ttest_rel(suspect_wm_scores, benign_wm_scores)
        t_vs_watermarked, p_vs_watermarked = ttest_rel(
            suspect_wm_scores, watermarked_wm_scores
        )

        # ── Detection logic ───────────────────────────────────────────────
        # A suspect is flagged as a watermark copy when:
        #   1. Its watermark scores are significantly HIGHER than the benign
        #      model's (p < 0.05, positive t-stat) — it recognises the pattern
        #   2. Its watermark scores are NOT significantly different from the
        #      watermarked model's (p > 0.05) — it matches the owner's model
        suspect_above_benign = (
            not math.isnan(t_vs_benign) and t_vs_benign > 0
            and not math.isnan(p_vs_benign) and p_vs_benign < 0.05
        )
        matches_watermarked = (
            math.isnan(p_vs_watermarked) or p_vs_watermarked > 0.05
        )
        detected = suspect_above_benign and matches_watermarked

        results = {
            # Classification confidence (reference)
            "benign_avg_confidence": benign_avg_conf,
            "watermarked_avg_confidence": watermarked_avg_conf,
            "suspect_avg_confidence": suspect_avg_conf,

            # Watermark head scores (primary signal)
            "benign_avg_wm_score": benign_avg_wm,
            "watermarked_avg_wm_score": watermarked_avg_wm,
            "suspect_avg_wm_score": suspect_avg_wm,
            "avg_dist_to_benign_wm": avg_dist_to_benign_wm,
            "avg_dist_to_watermarked_wm": avg_dist_to_watermarked_wm,

            # T-test results on watermark scores
            "t_stat_vs_benign": 0.0 if math.isnan(t_vs_benign) else t_vs_benign,
            "p_value_vs_benign": 1.0 if math.isnan(p_vs_benign) else p_vs_benign,
            "t_stat_vs_watermarked": 0.0 if math.isnan(t_vs_watermarked) else t_vs_watermarked,
            "p_value_vs_watermarked": 1.0 if math.isnan(p_vs_watermarked) else p_vs_watermarked,

            # Detection flags
            "suspect_above_benign": suspect_above_benign,
            "matches_watermarked": matches_watermarked,
            "detected": detected,

            # Raw score lists
            "benign_wm_scores": benign_wm_scores,
            "watermarked_wm_scores": watermarked_wm_scores,
            "suspect_wm_scores": suspect_wm_scores,
            "benign_confidences": benign_confs,
            "watermarked_confidences": watermarked_confs,
            "suspect_confidences": suspect_confs,
        }

        print(f"\n── Classification confidence ─────────────────────────────")
        print(f"benign avg confidence:           {benign_avg_conf:.4f}")
        print(f"watermarked avg confidence:      {watermarked_avg_conf:.4f}")
        print(f"suspect avg confidence:          {suspect_avg_conf:.4f}")
        print(f"\n── Watermark head scores (primary signal) ────────────────")
        print(f"benign avg wm score:             {benign_avg_wm:.4f}  (untrained → ~0.5)")
        print(f"watermarked avg wm score:        {watermarked_avg_wm:.4f}  (trained → ~1.0)")
        print(f"suspect avg wm score:            {suspect_avg_wm:.4f}")
        print(f"avg distance to benign wm:       {avg_dist_to_benign_wm:.4f}")
        print(f"avg distance to watermarked wm:  {avg_dist_to_watermarked_wm:.4f}")
        print(f"\n── Statistical tests on watermark scores ─────────────────")
        print(f"t-stat vs benign:                {results['t_stat_vs_benign']:+.4f}")
        print(f"p-value vs benign:               {results['p_value_vs_benign']:.4f}  {'✓ significant' if suspect_above_benign else '✗ not significant'}")
        print(f"p-value vs watermarked:          {results['p_value_vs_watermarked']:.4f}  {'✓ matches WM model' if matches_watermarked else '✗ does not match WM model'}")
        print(f"\n── Detection result ──────────────────────────────────────")
        print(f"suspect above benign:            {suspect_above_benign}")
        print(f"matches watermarked model:       {matches_watermarked}")
        print(f"DETECTED:                        {detected}")

        return results

    def verify_watermark(
        self,
        original_dataset: list,
        watermarked_graphs: list,
        chain_length: int
    ) -> bool:
        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        utility = UtilityFunctions()
        analyzer = GraphAnalyzer()

        indices = list(range(len(original_dataset)))
        rng.shuffle(indices)

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

            rng.shuffle(longest)
            selected_node = longest[0]
            expected_edge_node = selected_node[2]

            actual_length, tip = utility.get_dangling_chain_length(
                expected_edge_node, neighbors
            )

            if actual_length >= chain_length:
                verified += 1

        ratio = verified / len(watermarked_graphs) if watermarked_graphs else 0
        print(
            f"Watermark verification: {verified}/{len(watermarked_graphs)} "
            f"graphs confirmed ({ratio:.0%})"
        )
        return ratio > 0.8