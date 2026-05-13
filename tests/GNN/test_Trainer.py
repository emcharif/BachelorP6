import random

from src.GNN.Trainer import Trainer
from src.GNN.Classifier import Classifier
from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.inject_chain import inject_chain
from unittest.mock import patch

utils = UtilityFunctions()
analyzer = GraphAnalyzer()
rng = random.Random(1234)
dataset = utils.load_dataset(name="ENZYMES")

global_chain_length, _ = analyzer.get_longest_global_chain_length(dataset)
is_binary = utils.is_binary(dataset)
selected_graphs, _ = utils.graphs_to_watermark(dataset=dataset, rng=rng)

watermarked_graphs = [
    inject_chain(graph, global_chain_length, is_binary, rng, feature_mode="subtle")
    for graph in selected_graphs
]


def test_benign_and_wm_only_tested_on_clean_graphs():
    benign_trainer = Trainer(dataset=dataset)
    watermarked_trainer = Trainer(dataset=dataset, watermarked_graphs=watermarked_graphs)

    assert len(benign_trainer.train_loader) < len(watermarked_trainer.train_loader)
    assert len(benign_trainer.val_loader) == len(watermarked_trainer.val_loader)
    assert len(benign_trainer.test_loader) == len(watermarked_trainer.test_loader)


def test_train_returns_model():
    trainer = Trainer(dataset=dataset, epochs=1)
    with patch("torch.save"):
        model = trainer.train()

    assert model is not None
    assert isinstance(model, Classifier)


def test_train_returns_model_with_correct_dims_ENZYMES():
    trainer = Trainer(dataset=dataset, epochs=1)

    assert trainer.input_dim == 21
    assert trainer.hidden_dim == 128
    assert trainer.output_dim == 6


def test_train_returns_model_with_correct_dims_PROTEINS():
    dataset_proteins = utils.load_dataset(name="PROTEINS")
    trainer = Trainer(dataset=dataset_proteins, epochs=1)

    assert trainer.input_dim == 4
    assert trainer.hidden_dim == 128
    assert trainer.output_dim == 2