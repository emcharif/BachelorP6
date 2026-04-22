import torch
from torch_geometric.data import Data
from graph_analyzer import GraphAnalyzer


def make_graph(edges):
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = int(ei.max().item()) + 1
    return Data(x=torch.zeros(num_nodes, 2), edge_index=ei)

linear = make_graph([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]])

ring = make_graph([[0,1],[1,0],[1,2],[2,1],[2,0],[0,2]])

star = make_graph([[0,1],[1,0],[0,2],[2,0],[0,3],[3,0]])

single_edge = make_graph([[0,1],[1,0]])

analyzer = GraphAnalyzer()


def test_search_graph_returns_three_values():
    result = analyzer.search_graph(linear)
    assert len(result) == 3


def test_neighbors_contains_all_nodes():
    _, _, neighbors = analyzer.search_graph(linear)
    assert set(neighbors.keys()) == {0, 1, 2, 3}


def test_neighbors_are_symmetric():
    _, _, neighbors = analyzer.search_graph(linear)
    for node, nbrs in neighbors.items():
        for nbr in nbrs:
            assert node in neighbors[nbr]


def test_linear_chain_end_nodes_have_one_neighbor():
    _, _, neighbors = analyzer.search_graph(linear)
    assert len(neighbors[0]) == 1
    assert len(neighbors[3]) == 1


def test_linear_chain_middle_nodes_have_two_neighbors():
    _, _, neighbors = analyzer.search_graph(linear)
    assert len(neighbors[1]) == 2
    assert len(neighbors[2]) == 2


def test_ring_has_no_chain_starts():
    _, chain_starts, _ = analyzer.search_graph(ring)
    assert chain_starts == []


def test_linear_has_chain_starts():
    _, chain_starts, _ = analyzer.search_graph(linear)
    assert len(chain_starts) > 0


def test_single_edge_has_chain_starts():
    _, chain_starts, _ = analyzer.search_graph(single_edge)
    assert len(chain_starts) >= 1


def test_chain_starts_are_valid_node_ids():
    _, chain_starts, neighbors = analyzer.search_graph(linear)
    for node in chain_starts:
        assert node in neighbors


def test_star_center_has_three_neighbors():
    _, _, neighbors = analyzer.search_graph(star)
    assert len(neighbors[0]) == 3


def test_ring_all_nodes_have_two_neighbors():
    _, _, neighbors = analyzer.search_graph(ring)
    for node in neighbors:
        assert len(neighbors[node]) == 2


def test_get_global_chain_length_returns_int():
    result = analyzer.get_global_chain_length([linear])
    assert isinstance(result, int)


def test_get_global_chain_length_greater_than_zero():
    result = analyzer.get_global_chain_length([linear])
    assert result > 0


def test_get_global_chain_length_ring_returns_one():
    result = analyzer.get_global_chain_length([ring])
    assert result == 1


def test_get_global_chain_length_larger_graph_beats_smaller():
    small = make_graph([[0,1],[1,0],[1,2],[2,1]])                       
    large = make_graph([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]])    
    result_small = analyzer.get_global_chain_length([small])
    result_large = analyzer.get_global_chain_length([large])
    assert result_large > result_small


#test af get_dangling_chain_length
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
    
    length = analyzer.get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = analyzer.get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = analyzer.get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = analyzer.get_dangling_chain_length(startnode, neighbors)
    
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
    
    length = analyzer.get_dangling_chain_length(startnode, neighbors)
    
    assert length[0] == expected_length, f"Expected {expected_length}, got {length}"