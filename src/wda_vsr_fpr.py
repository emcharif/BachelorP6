import torch

from torch_geometric.data import Batch
from src.GNN.Classifier import Classifier


def _graph_vsr(model: Classifier, graph, target_label: int) -> float:
    """
    Computes VSR for a single graph (graph-level proxy):
    returns 1.0 if model predicts target_label, else 0.0.
    """
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list([graph])
        logits = model(batch)
        pred = logits.argmax(dim=1).item()
    if pred == target_label:
        return 1.0 
    else:
        return 0.0


def compute_wda_vsr_fpr(
    model: Classifier,
    watermarked_graphs: list,
    benign_graphs: list,
    target_label: int,
    num_classes: int,
) -> dict:
    """
    Computes WDA, VSR and FPR for a trigger-based baseline.

    Args:
        model:              Trained Classifier.
        watermarked_graphs: n+ watermarked suspect graphs.
        benign_graphs:      n- benign suspect graphs.
        target_label:       y* used during watermark injection.
        num_classes:        C — random-chance threshold is 1/C.
        seed:               Optional seed for any randomness (currently unused).

    Returns:
        wda: Watermark Detection Accuracy
        vsr: Verification Success Rate
        fpr: False Positive Rate
    """
    threshold = 1.0 / num_classes  # random-chance threshold

    suspects = []
    for graph in watermarked_graphs:
        suspects.append((graph, 1))
    for graph in benign_graphs:
        suspects.append((graph, 0))

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for graph, true_label in suspects:
        vsr = _graph_vsr(model, graph, target_label)
        flagged = int(vsr > threshold)

        if flagged == 1 and true_label == 1:
            true_positives += 1
        elif flagged == 0 and true_label == 0:
            true_negatives += 1
        elif flagged == 1 and true_label == 0:
            false_positives += 1
        else:
            false_negatives += 1

    wda = (true_positives + true_negatives) / len(suspects) 
    vsr = true_positives / len(watermarked_graphs)
    fpr = false_positives / len(benign_graphs)

    return {
        "wda": wda,
        "vsr": vsr,
        "fpr": fpr
    }