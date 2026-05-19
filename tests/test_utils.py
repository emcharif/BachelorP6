import torch
import random

from torch_geometric.data import Data
from src.utils import UtilityFunctions

utils = UtilityFunctions()

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

def test_dataset_is_binary():
    dataset = [Data(x = torch.tensor([[1], [0], [0], [1]], dtype=torch.float))]
    binary = utils.is_binary(dataset)
    assert binary

def test_dataset_is_not_binary():
    dataset = [Data(x = torch.tensor([[1.123], [9.32], [2.432], [-5.65]], dtype=torch.float))]
    binary = utils.is_binary(dataset)
    assert not binary

def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_5pct():
    rng = random.Random(1234)
    dataset = list(range(1, 21))
    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.05)    
    assert selected_graphs == [20]
    assert list(unselected_graphs) == [14, 5, 10, 17, 8, 7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_10pct():
    rng = random.Random(1234)
    dataset = list(range(1, 21))
    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.10)    
    assert selected_graphs == [20, 14]
    assert list(unselected_graphs) == [5, 10, 17, 8, 7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_20pct():
    rng = random.Random(1234)
    dataset = list(range(1, 21))
    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.20)    
    assert selected_graphs == [20, 14, 5, 10]
    assert list(unselected_graphs) == [17, 8, 7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graphs_to_watermark_always_returns_same_based_on_seeded_rng_30pct():
    rng = random.Random(1234)
    dataset = list(range(1, 21))
    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.30)    
    assert selected_graphs == [20, 14, 5, 10, 17, 8]
    assert list(unselected_graphs) == [7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graphs_to_watermark_returns_different_from_different_key_seeded_rng_30pct():
    rng = random.Random(4321)
    dataset = list(range(1, 21))
    selected_graphs, unselected_graphs = utils.graphs_to_watermark(dataset=dataset, rng=rng, percentage=0.30)    
    assert selected_graphs != [20, 14, 5, 10, 17, 8]
    assert list(unselected_graphs) != [7, 9, 18, 19, 6, 13, 16, 12, 11, 2, 3, 1, 4, 15]

def test_graph_to_watermark_same_label_return_correct_lengths():
    rng = random.Random(1234)
    dataset = utils.load_dataset(name="ENZYMES")
    selected_graphs, unselected_graphs = utils.graphs_to_watermark_same_label(dataset=dataset, graph_index=0, rng=rng)

    assert len(selected_graphs) == 30
    assert len(unselected_graphs) == 570

def test_graph_to_watermark_same_label_selected_graphs_contains_same_label():
    rng = random.Random(1234)
    dataset = utils.load_dataset(name="ENZYMES")
    selected_graphs, _ = utils.graphs_to_watermark_same_label(dataset=dataset, graph_index=0, rng=rng)

    test_label = selected_graphs[0].y

    for i in range(len(selected_graphs)):
        assert selected_graphs[i].y == test_label

def test_dif_watermarked_and_benign_graph_edges():

    benign_edges = [[1,1,2,2,3,3], [2,3,1,3,1,2]]
    watermarked_edges = [[1,1,2,2,3,3,3,4], [2,3,1,3,1,2,4,3]]

    delta_edges = utils.dif_watermarked_and_benign_graph_edges(benign_edges, watermarked_edges)

    assert delta_edges == ([3,4], [4,3])

def test_same_of_split_of_dataset_every_time():
    TRAIN_PCT = 0.70
    VAL_PCT = 0.15
    rng = random.Random(1234)
    dataset_example = [1,2,3,4,5,6,7,8,9,10]

    train_clean, val_clean, test_clean = UtilityFunctions().split_dataset(dataset_example, rng, TRAIN_PCT, VAL_PCT)
    train_clean1, val_clean1, test_clean1 = UtilityFunctions().split_dataset(dataset_example, rng, TRAIN_PCT, VAL_PCT)

    assert train_clean == train_clean1
    assert val_clean == val_clean1
    assert test_clean == test_clean1