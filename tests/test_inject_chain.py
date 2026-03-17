
import torch
from torch_geometric.data import Data
from src.undirected_graphs.inject_chain import inject_chain
from src.undirected_graphs.utils import UtilityFunctions
from src.undirected_graphs.graph_analyzer import GraphAnalyzer

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
    modified_graph = inject_chain(graph, chain_length)
    graph, chain_starts,edges, neighbors = GraphAnalyzer().search_graph(modified_graph)
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
    modified_graph = inject_chain(graph, chain_length)
    graph, chain_starts,edges, neighbors = GraphAnalyzer().search_graph(modified_graph)
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
    
    modified_graph = inject_chain(graph, chain_length)
    
    expected_new_nodes = chain_length - 3  # eksisterende chain er 3 lang
    assert modified_graph.num_nodes == original_num_nodes + expected_new_nodes, f"Expected {expected_new_nodes} new nodes, but got {modified_graph.num_nodes - original_num_nodes}"
   
