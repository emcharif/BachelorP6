import torch
from torch_geometric.data import Data
from src.graph_analyzer import GraphAnalyzer


def make_graph(edges):
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = int(ei.max().item()) + 1
    return Data(x=torch.zeros(num_nodes, 2), edge_index=ei)

linear = make_graph([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]])
ring = make_graph([[0,1],[1,0],[1,2],[2,1],[2,0],[0,2]])
star = make_graph([[0,1],[1,0],[0,2],[2,0],[0,3],[3,0]])
single_edge = make_graph([[0,1],[1,0]])

analyzer = GraphAnalyzer()


# --- search_graph ---

def test_search_graph_returns_two_values():
    result = analyzer.search_graph(linear)
    assert len(result) == 2


def test_neighbors_contains_all_nodes():
    _, neighbors = analyzer.search_graph(linear)
    assert set(neighbors.keys()) == {0, 1, 2, 3, 4}


def test_neighbors_are_symmetric():
    _, neighbors = analyzer.search_graph(linear)
    for node, nbrs in neighbors.items():
        for nbr in nbrs:
            assert node in neighbors[nbr]


def test_linear_chain_end_nodes_have_one_neighbor():
    _, neighbors = analyzer.search_graph(linear)
    assert len(neighbors[0]) == 1
    assert len(neighbors[4]) == 1


def test_linear_chain_middle_nodes_have_two_neighbors():
    _, neighbors = analyzer.search_graph(linear)
    assert len(neighbors[1]) == 2
    assert len(neighbors[2]) == 2


def test_ring_has_no_chain_starts():
    chain_starts, _ = analyzer.search_graph(ring)
    assert chain_starts == []


def test_linear_has_chain_starts():
    chain_starts, _ = analyzer.search_graph(linear)
    assert len(chain_starts) > 0


def test_single_edge_has_chain_starts():
    chain_starts, _ = analyzer.search_graph(single_edge)
    assert len(chain_starts) >= 1


def test_chain_starts_are_valid_node_ids():
    chain_starts, neighbors = analyzer.search_graph(linear)
    for node in chain_starts:
        assert node in neighbors


def test_star_center_has_three_neighbors():
    _, neighbors = analyzer.search_graph(star)
    assert len(neighbors[0]) == 3


def test_ring_all_nodes_have_two_neighbors():
    _, neighbors = analyzer.search_graph(ring)
    for node in neighbors:
        assert len(neighbors[node]) == 2


# --- get_longest_global_chain_length ---

def test_get_longest_global_chain_length_returns_int():
    result, _ = analyzer.get_longest_global_chain_length([linear])
    assert isinstance(result, int)


def test_get_longest_global_chain_length_greater_than_zero():
    result, _ = analyzer.get_longest_global_chain_length([linear])
    assert result > 0


def test_get_longest_global_chain_length_ring_returns_one():
    result, _ = analyzer.get_longest_global_chain_length([ring])
    assert result == 1


def test_get_longest_global_chain_length_larger_graph_beats_smaller():
    small = make_graph([[0,1],[1,0],[1,2],[2,1]])
    large = make_graph([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]])
    result_small, _ = analyzer.get_longest_global_chain_length([small])
    result_large, _ = analyzer.get_longest_global_chain_length([large])
    assert result_large > result_small


# --- get_shortest_global_chain_length ---

def test_get_shortest_global_chain_length_smaller_graph_beats_larger():
    small = make_graph([[0,1],[1,0],[1,2],[2,1],[3,0],[0,3],[1,3],[3,1]])
    large = make_graph([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3],[4,5],[5,4]])
    graph_index = analyzer.get_shortest_global_chain_length([small, large])
    assert graph_index == 0


def test_get_shortest_global_chain_length_return_correct_index():
    no_dangling = make_graph([[0,1],[1,0],[1,2],[2,1],[0,2],[2,0]])
    graph_index = analyzer.get_shortest_global_chain_length([no_dangling])
    assert graph_index == 0


def test_get_shortest_global_chain_length_contain_no_dangling():
    small = make_graph([[0,1],[1,0],[1,2],[2,1],[3,0],[0,3],[1,3],[3,1]])
    large = make_graph([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3],[4,5],[5,4]])
    graph_index = analyzer.get_shortest_global_chain_length([small, large])
    assert graph_index == 0


# --- get_dangling_chain_length ---

def test_get_dangling_chain_length():
    neighbors = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }
    startnode = 0
    length, _ = analyzer.get_dangling_chain_length(startnode, neighbors)
    assert length == 5


def test_get_dangling_chain_length_with_branching():
    neighbors = {
        0: [1],
        1: [0, 2, 5],
        2: [1, 3],
        3: [2, 4],
        4: [3],
        5: [1]
    }
    startnode = 0
    length, _ = analyzer.get_dangling_chain_length(startnode, neighbors)
    assert length == 1


def test_get_dangling_chain_length_from_cluster():
    neighbors = {
        0: [1, 7, 8, 9],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5]
    }
    startnode = 1
    length, _ = analyzer.get_dangling_chain_length(startnode, neighbors)
    assert length == 6


def test_get_dangling_chain_length_two_chains():
    neighbors = {
        0: [1, 4, 5, 6],
        1: [0, 2],
        2: [1, 3],
        3: [2],
        4: [0],
        5: [0],
        6: [0]
    }
    startnode = 1
    length, _ = analyzer.get_dangling_chain_length(startnode, neighbors)
    assert length == 3


def test_get_dangling_chain_length_single_node():
    neighbors = {
        0: [1, 2, 3],
        1: [0],
        2: [0],
        3: [0]
    }
    startnode = 1
    length, _ = analyzer.get_dangling_chain_length(startnode, neighbors)
    assert length == 1