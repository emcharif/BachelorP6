from utils import UtilityFunctions
from inject_chain import inject_chain
import torch

def test_inject_chain_increases_node_count():
    dataset = UtilityFunctions().load_dataset(name="IMDB-BINARY")
    graph = dataset[0]
    original_nodes = graph.x.shape[0]
    chain_length = 2
    modified = inject_chain(graph, chain_length, is_binary=True)
    assert modified.x.shape[0] >= original_nodes, "Node count should not decrease after injection"

def test_inject_chain_x_edge_index_consistent():
    dataset = UtilityFunctions().load_dataset(name="IMDB-BINARY")
    graph = dataset[0]
    modified = inject_chain(graph, chain_length=2, is_binary=True)
    assert modified.x.shape[0] == modified.edge_index.max().item() + 1, \
        "x rows should match number of unique nodes in edge_index"

def test_inject_chain_returns_clean_data():
    dataset = UtilityFunctions().load_dataset(name="IMDB-BINARY")
    graph = dataset[0]
    modified = inject_chain(graph, chain_length=2, is_binary=True)
    # Should not have stale num_nodes stored as attribute
    assert modified.x.shape[0] == modified.num_nodes, "num_nodes should match x"

def test_inject_chain_preserves_label():
    dataset = UtilityFunctions().load_dataset(name="IMDB-BINARY")
    graph = dataset[0]
    original_y = graph.y.clone()
    modified = inject_chain(graph, chain_length=2, is_binary=True)
    assert torch.equal(modified.y, original_y), "Label should not change after injection"