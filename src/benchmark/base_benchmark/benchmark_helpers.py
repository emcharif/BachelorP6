import hashlib
import torch
import torch.nn.functional as Function
import random
import copy

from src.inject_chain import inject_chain
from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from torch_geometric.loader import DataLoader

# Global seed offsets to ensure different random sequences for selection, injection, and verification steps
SEED_OFFSET_SELECT = 101
SEED_OFFSET_INJECT = 202
SEED_OFFSET_VERIFY = 303

def derive_seed(secret_key: str, dataset_name: str, repeat_idx: int) -> int:
    """
    Create a deterministic integer seed for one dataset/repeat combination.
    """
    seed_text = f"{secret_key}|{dataset_name}|repeat={repeat_idx}"
    seed_hash = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    seed_hex = seed_hash[:8]

    return int(seed_hex, 16)


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds for Python and PyTorch.
    This makes dataset splitting, model initialization, and training behavior
    more reproducible for a given experiment seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_dataset(dataset, seed: int, train_pct: float = 0.70, val_pct: float = 0.15) -> tuple [list, list, list]:
    """
    Split a dataset into train/validation/test splits based on the given percentages.
    The split is deterministic based on the provided seed.
        Args:
    - dataset: The full dataset to split (list of graphs).
    - seed: Random seed for reproducibility of the split.
    - train_pct: Percentage of the dataset to use for training (default 70%).
    - val_pct: Percentage of the dataset to use for validation (default 15%).

    Returns: 
        - Tuple containing the train, validation, and test splits as lists of graphs.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    train_size = int(train_pct * len(dataset))
    val_size = int(val_pct * len(dataset))

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_split = [copy.deepcopy(dataset[i]) for i in train_idx]
    val_split = [copy.deepcopy(dataset[i]) for i in val_idx]
    test_split = [copy.deepcopy(dataset[i]) for i in test_idx]

    return train_split, val_split, test_split

def tag_graph_for_watermark_head(graph, value: float):
    """
    Add a binary training target for the watermark head.
    """
    graph.is_watermarked = torch.tensor([value], dtype=torch.float)
    return graph

def build_watermarked_train_split(
    train_clean,
    watermark_pct: float,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    utility_functions: UtilityFunctions,
    feature_mode: str,
    use_watermark_head: bool,
) -> tuple [list, list, list]:
    """
    Build a watermarked training split by injecting the watermark pattern into a subset of the clean training graphs.

    Args: 
        - train_clean: List of clean training graphs.
        - watermark_pct: Percentage of training graphs to watermark.
        - target_chain_length: Length of the dangling chain to inject as the watermark pattern.
        - is_binary: Whether node/edge features are binary (affects how features are modified during injection).
        - seed: Base random seed for reproducibility.
        - utility_functions: Instance of UtilityFunctions for dataset manipulation.
        - feature_mode: "subtle" or "ood", determines how features are modified during injection.
        - use_watermark_head: Whether to add a binary target for the watermark head.

    Returns: 
        - tuple: Tuple containing the full watermarked training set, the list of watermarked graphs, and the list of clean graphs.
    """
    rng_select = random.Random(seed + SEED_OFFSET_SELECT)
    selected_graphs, unselected_graphs = utility_functions.graphs_to_watermark(
        dataset=train_clean,
        percentage=watermark_pct,
        rng=rng_select,
    )

    rng_inject = random.Random(seed + SEED_OFFSET_INJECT)

    watermarked_graphs = []
    for graph in selected_graphs:
        wm_graph = inject_chain(
            copy.deepcopy(graph),
            target_chain_length,
            is_binary,
            rng_inject,
            feature_mode=feature_mode,
        )

        if use_watermark_head:
            wm_graph = tag_graph_for_watermark_head(wm_graph, 1.0)

        watermarked_graphs.append(wm_graph)

    clean_graphs = []
    for graph in unselected_graphs:
        clean_graph = copy.deepcopy(graph)

        if use_watermark_head:
            clean_graph = tag_graph_for_watermark_head(clean_graph, 0.0)

        clean_graphs.append(clean_graph)

    watermarked_train = watermarked_graphs + clean_graphs

    return watermarked_train, watermarked_graphs, clean_graphs

def build_watermarked_train_split_same_label(
    train_clean,
    watermark_pct: float,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    utility_functions: UtilityFunctions,
    feature_mode: str,
    use_watermark_head: bool,
    graph_analyzer: GraphAnalyzer,
) -> tuple [list, list, list]:
    """
    Build a watermarked training split where all watermarked graphs share the same label.

    Args: 
        - train_clean: List of clean training graphs.
        - watermark_pct: Percentage of training graphs to watermark.
        - target_chain_length: Length of the dangling chain to inject as the watermark pattern.
        - is_binary: Whether node/edge features are binary (affects how features are modified during injection).
        - seed: Base random seed for reproducibility.
        - utility_functions: Instance of UtilityFunctions for dataset manipulation.
        - feature_mode: "subtle" or "ood", determines how features are modified during injection.
        - use_watermark_head: Whether to add a binary target for the watermark head.
        - graph_analyzer: Instance of GraphAnalyzer to analyze graph structures.

    Returns: 
        - tuple: Tuple containing the full watermarked training set, the list of watermarked graphs, and the list of clean graphs.
    """
    # Find the graph with the longest chain to use as the label anchor
    _, anchor_index = graph_analyzer.get_global_chain_length(train_clean)

    rng_select = random.Random(seed + SEED_OFFSET_SELECT)
    selected_graphs, unselected_graphs = utility_functions.graphs_to_watermark_same_label(
        dataset=train_clean,
        graph_index=anchor_index,
        percentage=watermark_pct,
        rng=rng_select,
    )

    rng_inject = random.Random(seed + SEED_OFFSET_INJECT)

    watermarked_graphs = []
    for graph in selected_graphs:
        wm_graph = inject_chain(
            copy.deepcopy(graph),
            target_chain_length,
            is_binary,
            rng_inject,
            feature_mode=feature_mode,
        )
        if use_watermark_head:
            wm_graph = tag_graph_for_watermark_head(wm_graph, 1.0)
        watermarked_graphs.append(wm_graph)

    clean_graphs = []
    for graph in unselected_graphs:
        clean_graph = copy.deepcopy(graph)
        if use_watermark_head:
            clean_graph = tag_graph_for_watermark_head(clean_graph, 0.0)
        clean_graphs.append(clean_graph)

    watermarked_train = watermarked_graphs + clean_graphs
    return watermarked_train, watermarked_graphs, clean_graphs

def build_verification_graphs(
    test_clean,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    verification_count: int,
    feature_mode: str,
) -> list:
    """
    Create a set of verification graphs by injecting the watermark pattern into a subset of the clean test graphs.
    Args:
        - test_clean: List of clean test graphs.
        - target_chain_length: Length of the dangling chain to inject as the watermark pattern.
        - is_binary: Whether node/edge features are binary (affects how features are modified during injection).
        - seed: Base random seed for reproducibility.
        - verification_count: Number of verification graphs to create.
        - feature_mode: "subtle" or "ood", determines how features are modified during injection.

    Returns: 
        - list: List of watermarked verification graphs.
    """
    rng = random.Random(seed + SEED_OFFSET_VERIFY)

    max_count = min(verification_count, len(test_clean))
    indices = list(range(len(test_clean)))
    rng.shuffle(indices)
    selected_indices = indices[:max_count]

    verification_graphs = []

    for idx in selected_indices:
        modified = inject_chain(
            copy.deepcopy(test_clean[idx]),
            target_chain_length,
            is_binary,
            rng,
            feature_mode=feature_mode,
        )
        verification_graphs.append(modified)

    return verification_graphs

def structurally_verify_watermark(watermarked_graphs, target_chain_length: int) -> bool:
    """
    Check if the watermark is present in the watermarked graphs by analyzing their structure.
    A graph is considered to contain the watermark if it has a dangling chain of at least the target length.
    
    Args:
        - watermarked_graphs: List of graphs that have been watermarked (i.e., had the dangling chain injected).
        - target_chain_length: The length of the dangling chain that was injected as the watermark pattern

    Returns: 
        - bool: True if the watermark is structurally verified in at least 80% of the watermarked graphs, False otherwise.
    """
    analyzer = GraphAnalyzer()
    verified = 0

    for graph in watermarked_graphs:
        _, chain_starts, neighbors = analyzer.search_graph(graph)

        lengths = []

        if len(chain_starts) != 0:
            for start in chain_starts:
                length, _ = analyzer.get_dangling_chain_length(start, neighbors)
                lengths.append(length)
        else:
            lengths.append(0)

        if max(lengths) >= target_chain_length:
            verified += 1

    ratio = verified / len(watermarked_graphs) if watermarked_graphs else 0
    print(
        f"Structural watermark verification: "
        f"{verified}/{len(watermarked_graphs)} graphs confirmed ({ratio:.0%})"
    )

    return ratio > 0.8

def collect_scores(model, graphs, batch_size: int):
    """
    Run the model on the given graphs and collect confidence and watermark scores.

    Args:
        - model: The GNN model to evaluate.
        - graphs: List of graphs to run through the model.
        - batch_size: Batch size for processing the graphs.

    Returns: 
        -dict: A dictionary containing average confidence, list of confidences, average watermark score, and list of watermark scores.
    """
    # batch graphs
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    # set model to eval mode
    model.eval()

    confidences = []
    watermark_scores = []

    # disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in loader:
            class_logits, wm_score = model(batch, return_watermark_score=True)

            probs = Function.softmax(class_logits, dim=1)
            conf = probs.max(dim=1).values

            confidences.extend(conf.cpu().tolist())
            watermark_scores.extend(wm_score.view(-1).cpu().tolist())

    return {
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "confidences": confidences,
        "avg_watermark_score": sum(watermark_scores) / len(watermark_scores) if watermark_scores else 0.0,
        "watermark_scores": watermark_scores,
    }

def test_models_on_verification_graphs(
    benign_model,
    watermarked_model,
    suspect_model,
    verification_graphs,
    batch_size: int,
) -> dict:
    """
    Run all three models on the verification graphs and collect confidence and watermark scores.

    Args:
        - benign_model: The model trained on clean data without watermarks.
        - watermarked_model: The model trained on watermarked data.
        - suspect_model: The model being evaluated for potential watermark presence.
        - verification_graphs: List of graphs with the watermark pattern injected for testing.
        - batch_size: Batch size for processing the graphs.

    Returns: 
        - dict: A dictionary containing average confidence and watermark scores for each model, as well as the full lists of scores.
    """
    benign = collect_scores(benign_model, verification_graphs, batch_size)
    watermarked = collect_scores(watermarked_model, verification_graphs, batch_size)
    suspect = collect_scores(suspect_model, verification_graphs, batch_size)

    return {
        "benign_avg_confidence": benign["avg_confidence"],
        "watermarked_avg_confidence": watermarked["avg_confidence"],
        "suspect_avg_confidence": suspect["avg_confidence"],

        "benign_confidences": benign["confidences"],
        "watermarked_confidences": watermarked["confidences"],
        "suspect_confidences": suspect["confidences"],

        "benign_avg_watermark_score": benign["avg_watermark_score"],
        "watermarked_avg_watermark_score": watermarked["avg_watermark_score"],
        "suspect_avg_watermark_score": suspect["avg_watermark_score"],

        "benign_watermark_scores": benign["watermark_scores"],
        "watermarked_watermark_scores": watermarked["watermark_scores"],
        "suspect_watermark_scores": suspect["watermark_scores"],
    }

def extract_signal_metrics(test_results: dict, prefix: str) -> dict:
    """
    Calculate signal metrics based on the collected confidence and watermark scores.
    
    Args:
        - test_results: Dictionary containing average confidence and watermark scores for benign, watermarked, and suspect models.
        - prefix: String prefix to add to the metric names (e.g., "overall", "watermark_head") for clarity in the results.

    Returns: 
        - dict: A dictionary containing calculated signal metrics comparing the suspect model to the benign and watermarked models.
    """
    benign_conf = test_results["benign_avg_confidence"]
    watermarked_conf = test_results["watermarked_avg_confidence"]
    suspect_conf = test_results["suspect_avg_confidence"]

    benign_wm = test_results["benign_avg_watermark_score"]
    watermarked_wm = test_results["watermarked_avg_watermark_score"]
    suspect_wm = test_results["suspect_avg_watermark_score"]

    return {
        f"{prefix}_suspect_minus_benign_confidence": suspect_conf - benign_conf,
        f"{prefix}_watermarked_minus_benign_confidence": watermarked_conf - benign_conf,
        f"{prefix}_suspect_minus_watermarked_confidence": suspect_conf - watermarked_conf,
        f"{prefix}_confidence_signal_positive_vs_benign": suspect_conf - benign_conf > 0,

        f"{prefix}_suspect_minus_benign_watermark_score": suspect_wm - benign_wm,
        f"{prefix}_watermarked_minus_benign_watermark_score": watermarked_wm - benign_wm,
        f"{prefix}_suspect_minus_watermarked_watermark_score": suspect_wm - watermarked_wm,
        f"{prefix}_watermark_head_signal_positive_vs_benign": suspect_wm - benign_wm > 0,
    }