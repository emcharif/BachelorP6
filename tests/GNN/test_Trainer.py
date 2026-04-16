import torch

from GNN.Trainer import Trainer
from torch_geometric.data import Data

dataset = [
    Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]), y=torch.tensor([0])),
    Data(x=torch.tensor([[0.0]]), edge_index=torch.tensor([[0], [0]]), y=torch.tensor([1])),
    Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]), y=torch.tensor([2])),
    Data(x=torch.tensor([[0.0]]), edge_index=torch.tensor([[0], [0]]), y=torch.tensor([3])),
    Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]), y=torch.tensor([4])),
    Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]), y=torch.tensor([5])),
    Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]), y=torch.tensor([6])),
]

def test_organize_dataset():

    trainer = Trainer(dataset=dataset)

    actual_train = []
    for graph in trainer.train_loader.dataset:
        actual_train.append(graph.y.item())

    actual_val = []
    for graph in trainer.val_loader.dataset:
        actual_val.append(graph.y.item())

    actual_test = []
    for graph in trainer.test_loader.dataset:
        actual_test.append(graph.y.item())

    assert actual_train == [3,2,1,0]
    assert actual_val == [5]
    assert actual_test == [6,4]