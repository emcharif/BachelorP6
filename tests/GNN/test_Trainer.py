import random

from src.GNN.Trainer import Trainer
from src.GNN.Classifier import Classifier
from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.inject_chain import inject_chain
from unittest.mock import patch

def helper(dataset_name: str=None):

    TRAIN_PCT = 0.70
    VAL_PCT = 0.15

    utils = UtilityFunctions()
    analyzer = GraphAnalyzer()
    rng = random.Random(1234)

    dataset = utils.load_dataset(name=dataset_name)

    global_chain_length, graph_index = analyzer.get_longest_global_chain_length(dataset)
    is_binary = utils.is_binary(dataset)
    selected_graphs, _ = utils.graphs_to_watermark(dataset=dataset, rng=rng)

    train_clean, val_clean, test_clean = utils.split_dataset(dataset, rng, TRAIN_PCT, VAL_PCT)

    selected_graphs, unselected_graphs = utils.graphs_to_watermark_same_label(dataset=list(train_clean), graph_index=graph_index, rng=rng)

    watermarked_graphs = []
    for graph in selected_graphs:
        modified_graph = inject_chain(graph=graph, target_chain_length=global_chain_length, is_binary=is_binary, rng=rng, feature_mode="subtle")
        watermarked_graphs.append(modified_graph)

    watermarked_train_split = watermarked_graphs + unselected_graphs

    return train_clean, val_clean, test_clean, watermarked_train_split


def test_train_returns_model():
    
    train_clean, val_clean, test_clean, _ = helper(dataset_name = "ENZYMES")

    trainer = Trainer(train_dataset=train_clean, val_dataset=val_clean, test_dataset=test_clean, epochs=1)
    with patch("torch.save"):
        model = trainer.train()

    assert model is not None
    assert isinstance(model, Classifier)


def test_train_returns_model_with_correct_dims_ENZYMES():

    train_clean, val_clean, test_clean, _ = helper(dataset_name = "ENZYMES")

    trainer = Trainer(train_dataset=train_clean, val_dataset=val_clean, test_dataset=test_clean, epochs=1)

    assert trainer.input_dim == 21
    assert trainer.hidden_dim == 128
    assert trainer.output_dim == 6


def test_train_returns_model_with_correct_dims_PROTEINS():

    train_clean, val_clean, test_clean, _ = helper(dataset_name = "PROTEINS")

    trainer = Trainer(train_dataset=train_clean, val_dataset=val_clean, test_dataset=test_clean, epochs=1)

    assert trainer.input_dim == 4
    assert trainer.hidden_dim == 128
    assert trainer.output_dim == 2