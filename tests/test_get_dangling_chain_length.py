import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.undirected_graphs.utils import UtilityFunctions

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
    
    assert length == expected_length, f"Expected {expected_length}, got {length}"

def test_get_dangling_chain_length_with_branching():
    neighbors = {
        0: [1],
        1: [0, 2, 5],  # Branching at node 1
        2: [1, 3],
        3: [2, 4],
        4: [3],
        5: [1]  # Branching node
    }
    
    startnode = 0
    expected_length = 1  # The chain is 0 -> 1, but it branches at node 1
    
    length = UtilityFunctions().get_dangling_chain_length(startnode, neighbors)
    
    assert length == expected_length, f"Expected {expected_length}, got {length}"