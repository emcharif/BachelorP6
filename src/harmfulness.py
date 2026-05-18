import torch

from src.GNN.Classifier import Classifier
from torch_geometric.data import Batch

def compute_harmlessness(
    benign_model: Classifier,
    watermarked_model: Classifier,
    watermarked_graphs: list,
) -> dict:
    """
    Computes graph-level harmlessness metrics H and H_hat.

    H    = fraction of watermarked graphs where watermarked model predicts wrong label
    H_hat = watermark-induced change in error rate (watermarked model wrong - benign model wrong)
    """
    benign_model.eval()
    watermarked_model.eval()

    errors_wm = 0
    errors_benign = 0

    with torch.no_grad():
        for graph in watermarked_graphs:
            true_label = graph.y.item()
            batch = Batch.from_data_list([graph])

            pred_wm = watermarked_model(batch).argmax(dim=1).item()
            pred_benign = benign_model(batch).argmax(dim=1).item()

            if pred_wm != true_label:
                errors_wm += 1
            if pred_benign != true_label:
                errors_benign += 1

    H     = errors_wm / len(watermarked_graphs)
    H_hat = (errors_wm - errors_benign) / len(watermarked_graphs)

    return {
        "H":     H,
        "H_hat": H_hat,
    }