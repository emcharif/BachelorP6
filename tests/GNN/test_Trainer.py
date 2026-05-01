import random

from GNN.Trainer import Trainer
from GNN.Classifier import Classifier
from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from inject_chain import inject_chain
from load_model import ModelLoader
from unittest.mock import patch

utils = UtilityFunctions()
analyzer = GraphAnalyzer()
model_loader = ModelLoader()
rng = random.Random(1234)
dataset = utils.load_dataset(name = "ENZYMES")

global_chain_length, _ = analyzer.get_global_chain_length(dataset)
is_binary = utils.is_binary(dataset)
selected_graphs, _ = utils.graphs_to_watermark(dataset=dataset, rng=rng)

watermarked_graphs = [inject_chain(graph, global_chain_length, is_binary, rng) for graph in selected_graphs]

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
    dataset = utils.load_dataset(name = "PROTEINS")

    trainer = Trainer(dataset=dataset, epochs=1)

    assert trainer.input_dim == 4
    assert trainer.hidden_dim == 128
    assert trainer.output_dim == 2

def test_evaluate_returns_0_to_1():
    trainer = Trainer(dataset=dataset, epochs=1)
    with patch("torch.save"):
        trainer.train()
    acc = trainer.evaluate(trainer.test_loader)

    assert 0.0 <= acc <= 1.0

def test_make_preds_and_confs_for_all_graphs_inputted():
    trainer = Trainer(dataset=dataset, epochs=1)
    with patch("torch.save"):
        model = trainer.train()
        predictions, confidences = trainer.get_predictions(model=model, dataset=watermarked_graphs)

    assert len(predictions) == len(watermarked_graphs)
    assert len(confidences) == len(watermarked_graphs)

def test_preds_only_guessing_within_label_range():
    trainer = Trainer(dataset=dataset, epochs=1)
    with patch("torch.save"):
        model = trainer.train()
        predictions, _ = trainer.get_predictions(model=model, dataset=watermarked_graphs)

    for pred in predictions:
        assert pred in range(trainer.output_dim)

def test_confs_withing_0_to_1():
    trainer = Trainer(dataset=dataset, epochs=1)
    with patch("torch.save"):
        model = trainer.train()
        _, confidences = trainer.get_predictions(model=model, dataset=watermarked_graphs)

    for conf in confidences:
        assert 0.0 <= conf <= 1.0

def test_is_model_trained_on_watermarked_dataset_returns_bool():
    benign_trainer = Trainer(dataset=dataset, epochs=1)
    watermarked_trainer = Trainer(dataset=dataset, epochs=1, watermarked_graphs=watermarked_graphs)

    with patch("torch.save"):
        benign_model = benign_trainer.train()
        watermarked_model = watermarked_trainer.train()

    path = "models/ENZYMES/watermarked_model.pth" #just test model
    with open(path, "rb") as file:
        file_bytes = file.read()

    suspect_model = model_loader.load_model(file_bytes=file_bytes)

    result = benign_trainer.is_model_trained_on_watermarked_dataset(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=suspect_model,
        original_dataset=dataset,
        watermarked_graphs=watermarked_graphs
    )

    print(type(result))
    print(result)
    assert isinstance(result, bool)

def test_verify_watermark_returns_bool():
    trainer = Trainer(dataset=dataset, epochs=1)
    result = trainer.verify_watermark(original_dataset=dataset, watermarked_graphs=watermarked_graphs, chain_length=global_chain_length)

    assert isinstance(result, bool)