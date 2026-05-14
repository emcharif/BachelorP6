import math
import torch

from scipy.stats import ttest_rel
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from src.GNN.Classifier import Classifier

class Evaluator:

    def evaluate(self, loader: DataLoader) -> float:
        """
        Computes classification accuracy over a DataLoader.

        Handles both plain models and models with use_watermark_head=True,
        where forward() returns a (class_logits, wm_score) tuple — in that
        case only the class_logits are used for evaluation.

        Args:
            loader: A PyG DataLoader yielding batched graphs with ground-truth
                    labels in batch.y.

        Returns:
            Accuracy as a float in [0, 1], or 0.0 if the loader is empty.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

        if total > 0:
            return correct / total
        else:
            return 0.0

    def get_predictions(self, model: Classifier, dataset: list[Data]) -> list[float]:
        """
        Runs inference on a list of graphs and returns the model's confidence in terms of softmax for each prediction.

        Args:
            model:   The trained Classifier to run inference with.
            dataset: A list of PyG Data objects to predict on.

        Returns:
            A list of confidence scores (max softmax probability) for each
            graph, in the same order as the input dataset.
        """
        model.eval()
        confidences = []
        with torch.no_grad():
            for graph in dataset:
                batch = Batch.from_data_list([graph])
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                conf = probs.max(dim=1).values.item()
                confidences.append(conf)
        return confidences
    
    def test_models_with_watermark(self, benign_model: Classifier, watermarked_model: Classifier, suspect_model: Classifier, watermarked_graphs: list[Data]) -> dict:
        """
        Compares a suspect model's confidence scores against a benign and a
        watermarked model over a set of unseen watermarked graphs.

        For each graph, the max softmax confidence is collected from all three
        models. A paired t-test is then run to determine whether the suspect's
        confidences are statistically closer to the benign or the watermarked
        model, which serves as an indicator of whether the suspect was trained
        on watermarked data.

        Args:
            benign_model:       A Classifier trained on the clean dataset
                                without any watermark.
            watermarked_model:  A Classifier trained on the watermarked dataset.
            suspect_model:      A Classifier of unknown provenance to be tested.
            watermarked_graphs: A list of PyG Data objects with injected watermark
                                chains, used as the test set for all three models.

        Returns:
            A dict containing:
                - benign_avg_confidence:        Mean confidence of the benign model.
                - watermarked_avg_confidence:   Mean confidence of the watermarked model.
                - suspect_avg_confidence:       Mean confidence of the suspect model.
                - avg_distance_to_benign:       Mean absolute difference between
                                                suspect and benign confidences.
                - avg_distance_to_watermarked:  Mean absolute difference between
                                                suspect and watermarked confidences.
                - t_stat_vs_benign:             Paired t-statistic vs benign model.
                - p_value_vs_benign:            Paired t-test p-value vs benign model.
                - t_stat_vs_watermarked:        Paired t-statistic vs watermarked model.
                - p_value_vs_watermarked:       Paired t-test p-value vs watermarked model.
                - benign_confidences:           Per-graph confidence list for benign model.
                - watermarked_confidences:      Per-graph confidence list for watermarked model.
                - suspect_confidences:          Per-graph confidence list for suspect model.
            """


        benign_confs = self.get_predictions(benign_model, watermarked_graphs)
        watermarked_confs = self.get_predictions(watermarked_model, watermarked_graphs)
        suspect_confs = self.get_predictions(suspect_model, watermarked_graphs)

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
        }

        return results