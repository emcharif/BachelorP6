from src.utils import UtilityFunctions
from src.inject_chain import inject_chain
import torch
import random

rng = random.Random(1234)
utils = UtilityFunctions()


def test_inject_chain_increases_node_count():
    dataset = utils.load_dataset(name="ENZYMES")
    graph = dataset[0]
    original_num_nodes = graph.x.shape[0]
    modified = inject_chain(graph, target_chain_length=2, is_binary=False, rng=rng, feature_mode="subtle")
    assert modified.x.shape[0] >= original_num_nodes, "Node count should not decrease after injection"

def test_inject_chain_x_edge_index_consistent():
    dataset = utils.load_dataset(name="IMDB-BINARY")
    graph = dataset[0]
    modified = inject_chain(graph, target_chain_length=2, is_binary=True, rng=rng, feature_mode="subtle")
    assert modified.x.shape[0] == modified.edge_index.max().item() + 1, \
        "x rows should match number of unique nodes in edge_index"

def test_inject_chain_returns_clean_data():
    dataset = utils.load_dataset(name="IMDB-BINARY")
    graph = dataset[0]
    modified = inject_chain(graph, target_chain_length=2, is_binary=True, rng=rng, feature_mode="subtle")
    assert modified.x.shape[0] == modified.num_nodes, "num_nodes should match x"

def test_inject_chain_preserves_label():
    dataset = utils.load_dataset(name="IMDB-BINARY")
    graph = dataset[0]
    original_y = graph.y.clone()
    modified = inject_chain(graph, target_chain_length=2, is_binary=True, rng=rng, feature_mode="subtle")
    assert torch.equal(modified.y, original_y), "Label should not change after injection"