import torch

from torch_geometric.data import Data
from utils import UtilityFunctions

utils = UtilityFunctions()

#========================load_dataset()================
def test_load_dataset():
    dataset = utils.load_dataset(name = "ENZYMES")

    assert dataset[0].x is not None
    assert dataset[0].edge_index is not None
    assert dataset[0].y is not None 

def test_load_dataset_no_node_features():
    dataset = utils.load_dataset(name="MUTAG")

    assert dataset[0].x[0][0] == 1

#=================get_dangling_chain_length()====================
def test_get_dangling_chain_length():
    neighbors = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }
    
    startnode = 0
    expected_length = 5  # The chain is 0 -> 1 -> 2 -> 3 -> 4
    
    length = utils.get_dangling_chain_length(startnode, neighbors)
    
    assert length[0] == expected_length, f"Expected {expected_length}, got {length}"

def test_get_dangling_chain_length_with_branching():
    neighbors = {
        0: [1],
        1: [0, 2, 5],  # Branching at node 1
        2: [1, 3],
        3: [2, 4],
        4: [3],
        5: [1]  
    }
    
    startnode = 0
    expected_length = 1  # The chain is 0 -> 1, but it branches at node 1
    
    length = utils.get_dangling_chain_length(startnode, neighbors)
    
    assert length[0] == expected_length, f"Expected {expected_length}, got {length}"

def test_get_dangling_chain_length_from_cluster():
    neighbors = {
        0: [1,7,8,9],  # Cluster starts at node 0
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],  
        5: [4, 6],
        6: [5]
    }
    
    startnode = 1
    expected_length = 6  # The chain is 1 -> 2 -> 3 -> 4 -> 5 -> 6
    
    length = utils.get_dangling_chain_length(startnode, neighbors)
    
    assert length[0] == expected_length, f"Expected {expected_length}, got {length}"

def test_get_dangling_chain_length_two_chains():
    neighbors = {
        0: [1, 4, 5, 6],  # cluster
        1: [0, 2],
        2: [1, 3],
        3: [2],            
        4: [0],            
        5: [0],
        6: [0]
    }
    
    startnode = 1
    expected_length = 3
    
    length = utils.get_dangling_chain_length(startnode, neighbors)
    
    assert length[0] == expected_length, f"Expected {expected_length}, got {length}"

def test_get_dangling_chain_length_single_node():
    neighbors = {
        0: [1, 2, 3],
        1: [0],  
        2: [0],
        3: [0]
    }
    
    startnode = 1
    expected_length = 1
    
    length = utils.get_dangling_chain_length(startnode, neighbors)
    
    assert length[0] == expected_length, f"Expected {expected_length}, got {length}"

#=================is_binary()============================================
def test_dataset_is_binary():
    dataset = [Data(x = torch.tensor([[1], [0], [0], [1]], dtype=torch.float))]
    
    binary = utils.is_binary(dataset)

    assert binary == True

def test_dataset_is_not_binary():
    dataset = [Data(x = torch.tensor([[1.123], [9.32], [2.432], [-5.65]], dtype=torch.float))]
    
    binary = utils.is_binary(dataset)

    assert binary == False

#=================dif_watermarked_and_benign_graph_edges()================
def test_dif_watermarked_and_benign_graph_edges():

    a = [[1,1,2,2,3,3], [2,3,1,3,1,2]]
    b = [[1,1,2,2,3,3,3,4], [2,3,1,3,1,2,4,3]]

    x, y, delta = utils.dif_watermarked_and_benign_graph_edges(a, b)

    assert x == a
    assert y == b
    assert delta == [[3,4], [4,3]]