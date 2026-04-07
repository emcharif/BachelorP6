import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import UtilityFunctions

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
    
    length = UtilityFunctions().get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = UtilityFunctions().get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = UtilityFunctions().get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = UtilityFunctions().get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = UtilityFunctions().get_dangling_chain_length(startnode, neighbors)
    
    assert length[0] == expected_length, f"Expected {expected_length}, got {length}"