
import torch
from torch_geometric.data import Data
from src.inject_chain import inject_chain
from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer

def test_inject_chain_baseCase():
    # Create a simple graph for testing
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],  # source
        [1, 2, 3, 4, 5, 0, 1, 4] # target
         ], dtype=torch.long)
    graph = Data(edge_index=edge_index, num_nodes=6)

    # Define the chain length
    chain_length = 5

    # Call the inject_chain function
    modified_graph = inject_chain(graph, chain_length, is_binary=False)
    graph, chain_starts,edges, neighbors, chain_lengths = GraphAnalyzer().search_graph(modified_graph)
    dangling_nodes = chain_starts
        # Check if the chain was injected correctly
    lengths = []
    for d in dangling_nodes:
        length, edge_node = UtilityFunctions().get_dangling_chain_length(d, neighbors)
        lengths.append(length)

    # Assert that the longest chain length is equal to the specified chain length
    assert max(lengths) == chain_length, f"Expected chain length of {chain_length}, but got {max(lengths)}"


def test_inject_chain_noDanglingNodes():
    # Create a simple graph with no dangling nodes
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],  # source
        [1, 2, 0, 2, 0, 1] # target
         ], dtype=torch.long)
    graph = Data(edge_index=edge_index, num_nodes=3)

    # Define the chain length
    chain_length = 5

    # Call the inject_chain function
    modified_graph = inject_chain(graph, chain_length, is_binary=False)
    graph, chain_starts,edges, neighbors, chain_lengths = GraphAnalyzer().search_graph(modified_graph)
    dangling_nodes = chain_starts
    
    # Check if the chain was injected correctly
    lengths = []
    for d in dangling_nodes:
        length, edge_node = UtilityFunctions().get_dangling_chain_length(d, neighbors)
        lengths.append(length)
    
    assert max(lengths) == chain_length

def test_inject_chain_correctNumberOfNodesAdded():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 4]
    ], dtype=torch.long)
    graph = Data(edge_index=edge_index, num_nodes=6)
    
    original_num_nodes = graph.num_nodes
    chain_length = 5
    
    modified_graph = inject_chain(graph, chain_length, is_binary=False)
    
    expected_new_nodes = chain_length - 3  # eksisterende chain er 3 lang
    assert modified_graph.num_nodes == original_num_nodes + expected_new_nodes, f"Expected {expected_new_nodes} new nodes, but got {modified_graph.num_nodes - original_num_nodes}"
   
def test_inject_chain_with_node_attributes():
    # Graf med node attributter — 3 noder, 3 features per node
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 4]
    ], dtype=torch.long)
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=torch.float)
    graph = Data(edge_index=edge_index, x=x, num_nodes=6)

    chain_length = 5
    modified_graph = inject_chain(graph, chain_length, is_binary=False)

    # Verificer at antal node feature rækker matcher num_nodes
    assert modified_graph.x.shape[0] == modified_graph.num_nodes, \
        f"Expected {modified_graph.num_nodes} node feature rows, got {modified_graph.x.shape[0]}"
    
    # Verificer at feature dimensionen ikke er ændret
    assert modified_graph.x.shape[1] == 3, \
        f"Expected 3 features per node, got {modified_graph.x.shape[1]}"


def test_inject_chain_with_binary_node_attributes():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 4]
    ], dtype=torch.long)
    x = torch.tensor([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
    ], dtype=torch.float)
    graph = Data(edge_index=edge_index, x=x, num_nodes=6)

    chain_length = 5
    modified_graph = inject_chain(graph, chain_length, is_binary=True)

    # Verificer at nye noder har præcis samme features som edge_node — ingen deviationer
    original_num_nodes = 6
    for new_node_idx in range(original_num_nodes, modified_graph.num_nodes):
        new_features = modified_graph.x[new_node_idx]
        # Binære features må kun indeholde 0 eller 1
        assert torch.all((new_features == 0) | (new_features == 1)), \
            f"Binary features should only contain 0 or 1, got {new_features}"


def test_inject_chain_with_edge_attributes():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 4]
    ], dtype=torch.long)
    edge_attr = torch.ones(8, 4)  # 8 kanter, 4 features per kant
    graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=6)

    chain_length = 5
    modified_graph = inject_chain(graph, chain_length, is_binary=False)

    # Verificer at antal edge feature rækker matcher antal kanter
    num_edges = modified_graph.edge_index.shape[1]
    assert modified_graph.edge_attr.shape[0] == num_edges, \
        f"Expected {num_edges} edge feature rows, got {modified_graph.edge_attr.shape[0]}"

    # Verificer at edge feature dimensionen ikke er ændret
    assert modified_graph.edge_attr.shape[1] == 4, \
        f"Expected 4 features per edge, got {modified_graph.edge_attr.shape[1]}"


def test_inject_chain_deviations_within_range():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 4]
    ], dtype=torch.long)
    x = torch.tensor([
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
    ], dtype=torch.float)
    graph = Data(edge_index=edge_index, x=x, num_nodes=6)

    chain_length = 5
    original_num_nodes = 6
    modified_graph = inject_chain(graph, chain_length, is_binary=False)

    # Verificer at nye node features ligger inden for 97%-102% af original
    for new_node_idx in range(original_num_nodes, modified_graph.num_nodes):
        new_features = modified_graph.x[new_node_idx]
        for feat_idx in range(new_features.shape[0]):
            original_value = x[0][feat_idx].item()
            new_value = new_features[feat_idx].item()
            assert original_value * 0.95 <= new_value <= original_value * 1.04

def test_inject_chain_edge_attribute_and_node_attribute():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 4]
    ], dtype=torch.long)
    x = torch.tensor([
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
    ], dtype=torch.float)
    edge_attr = torch.ones(8, 4)  # 8 kanter, 4 features per kant
    graph = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=6)

    chain_length = 5
    original_num_nodes = 6
    modified_graph = inject_chain(graph, chain_length, is_binary=False)

    # Verificer at nye node features ligger inden for 97%-102% af original
    for new_node_idx in range(original_num_nodes, modified_graph.num_nodes):
        new_features = modified_graph.x[new_node_idx]
        for feat_idx in range(new_features.shape[0]):
            original_value = x[0][feat_idx].item()
            new_value = new_features[feat_idx].item()
            assert original_value * 0.96 <= new_value <= original_value * 1.04

def test_inject_chain_is_deterministic():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 4, 1, 4, 5],
        [1, 2, 3, 4, 5, 0, 1, 4]
    ], dtype=torch.long)
    
    graph1 = Data(edge_index=edge_index.clone(), num_nodes=6)
    graph2 = Data(edge_index=edge_index.clone(), num_nodes=6)

    result1 = inject_chain(graph1, chain_length=5, is_binary=False)
    result2 = inject_chain(graph2, chain_length=5, is_binary=False)

    assert torch.equal(result1.edge_index, result2.edge_index), "Same input should give same output"
    assert result1.num_nodes == result2.num_nodes