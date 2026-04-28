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


#=================is_binary()============================================
def test_dataset_is_binary():
    dataset = [Data(x = torch.tensor([[1], [0], [0], [1]], dtype=torch.float))]
    
    binary = utils.is_binary(dataset)

    assert binary

def test_dataset_is_not_binary():
    dataset = [Data(x = torch.tensor([[1.123], [9.32], [2.432], [-5.65]], dtype=torch.float))]
    
    binary = utils.is_binary(dataset)

    assert not binary

#=================dif_watermarked_and_benign_graph_edges()================
def test_dif_watermarked_and_benign_graph_edges():

    a = [[1,1,2,2,3,3], [2,3,1,3,1,2]]
    b = [[1,1,2,2,3,3,3,4], [2,3,1,3,1,2,4,3]]

    x, delta = utils.dif_watermarked_and_benign_graph_edges(a, b)

    assert x == a
    assert delta == [[3,4], [4,3]]