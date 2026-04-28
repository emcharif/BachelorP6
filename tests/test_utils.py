import torch
import random

from torch_geometric.data import Data
from utils import UtilityFunctions

utils = UtilityFunctions()

#========================load_dataset()================
def test_load_dataset_ENZYMES():
    dataset = utils.load_dataset(name = "ENZYMES")

    assert dataset[0].x is not None
    assert dataset[0].edge_index is not None
    assert dataset[0].y is not None 

def test_load_dataset_no_node_features():
    dataset = utils.load_dataset(name="MUTAG")

    assert dataset[0].x[0][0] == 1

def test_load_dataset_IMDB_BINARY():
    dataset = UtilityFunctions().load_dataset(name="IMDB-BINARY")
    assert len(dataset) > 0, "Dataset should not be empty"
    for graph in dataset:
        assert graph.x is not None, "Node features should not be None"
        assert graph.x.shape[0] == graph.num_nodes, "Node features should have the same number of rows as nodes"


#===========select_dangling_node()============================
def test_select_dangling_node_always_returns_same_based_on_seeded_rng():
    rng = random.Random(1234)
    longest_dangling_node_candidates = [[1,4,3],[2,4,1],[10,4,8],[100,4,98]]

    selected_dangling_node = utils.select_dangling_node(dangling_chain=longest_dangling_node_candidates, rng=rng)

    assert selected_dangling_node == [2,4,1]


#=================is_binary()============================================
def test_dataset_is_binary():
    dataset = [Data(x = torch.tensor([[1], [0], [0], [1]], dtype=torch.float))]
    
    binary = utils.is_binary(dataset)

    assert binary

def test_dataset_is_not_binary():
    dataset = [Data(x = torch.tensor([[1.123], [9.32], [2.432], [-5.65]], dtype=torch.float))]
    
    binary = utils.is_binary(dataset)

    assert not binary

#=======graphs_to_watermark()=================
def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_5pct():
    rng = random.Random(1234)

    dataset = []
    for i in range(1, 21):
        dataset.append(i)

    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.05)    

    assert selected_graphs == [20]
    assert unselected_graphs == [14, 5, 10, 17, 8, 7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_10pct():
    rng = random.Random(1234)

    dataset = []
    for i in range(1, 21):
        dataset.append(i)

    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.10)    

    assert selected_graphs == [20, 14]
    assert unselected_graphs == [5, 10, 17, 8, 7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_20pct():
    rng = random.Random(1234)

    dataset = []
    for i in range(1, 21):
        dataset.append(i)

    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.20)    

    assert selected_graphs == [20, 14, 5, 10]
    assert unselected_graphs == [17, 8, 7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_30pct():
    rng = random.Random(1234)

    dataset = []
    for i in range(1, 21):
        dataset.append(i)

    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.30)    

    assert selected_graphs == [20, 14, 5, 10, 17, 8]
    assert unselected_graphs == [7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]


#=================dif_watermarked_and_benign_graph_edges()================
def test_dif_watermarked_and_benign_graph_edges():

    a = [[1,1,2,2,3,3], [2,3,1,3,1,2]]
    b = [[1,1,2,2,3,3,3,4], [2,3,1,3,1,2,4,3]]

    x, delta = utils.dif_watermarked_and_benign_graph_edges(a, b)

    assert x == a
    assert delta == [[3,4], [4,3]]