import torch
from unittest.mock import patch
from torch_geometric.data import Data

from src.undirected_graphs.GNN.Trainer import Trainer

# ================= CONSTANTS ============================================
DATASET_NAME = "MUTAG"
INPUT_DIM    = 7
HIDDEN_DIM   = 64
OUTPUT_DIM   = 2
NUM_NODES    = 5
NUM_GRAPHS   = 20


# ================= HELPERS ==============================================
def make_mock_graph(num_nodes=NUM_NODES, input_dim=INPUT_DIM, num_edges=4, label=0):
    """Returns a mock PyG Data object representing a single graph."""
    data = Data()
    data.x          = torch.randn(num_nodes, input_dim)
    data.edge_index = torch.randint(0, num_nodes, (2, num_edges))
    data.y          = torch.tensor([label])
    data.num_nodes  = num_nodes
    return data


def make_mock_dataset(num_graphs=NUM_GRAPHS, input_dim=INPUT_DIM):
    """Returns a list of mock graphs with alternating labels."""
    dataset = []
    for i in range(num_graphs):
        y = i % 2
        dataset.append(make_mock_graph(input_dim=input_dim, label=y))

    return dataset


def make_trained_trainer(num_graphs=NUM_GRAPHS, epochs=2):
    """Returns a Trainer that has been loaded and trained on mock data."""
    trainer = Trainer(dataset_name=DATASET_NAME, hidden_dim=HIDDEN_DIM, epochs=epochs)
    dataset = make_mock_dataset(num_graphs=num_graphs)
    with patch("src.undirected_graphs.GNN.Trainer.load_dotenv"), \
         patch("src.undirected_graphs.GNN.Trainer.os.getenv", return_value="test_seed"):
        trainer.load_data(dataset=dataset)
    trainer.train()
    return trainer


# ================= LOAD DATA TESTS ======================================
def test_load_data_creates_all_loaders():
    trainer = Trainer(dataset_name=DATASET_NAME)
    dataset = make_mock_dataset()

    with patch("src.undirected_graphs.GNN.Trainer.load_dotenv"), \
         patch("src.undirected_graphs.GNN.Trainer.os.getenv", return_value="test_seed"):
        trainer.load_data(dataset=dataset)

    assert trainer.train_loader is not None
    assert trainer.val_loader   is not None
    assert trainer.test_loader  is not None


def test_load_data_sets_input_and_output_dim():
    trainer = Trainer(dataset_name=DATASET_NAME)
    dataset = make_mock_dataset(input_dim=INPUT_DIM)

    with patch("src.undirected_graphs.GNN.Trainer.load_dotenv"), \
         patch("src.undirected_graphs.GNN.Trainer.os.getenv", return_value="test_seed"):
        trainer.load_data(dataset=dataset)

    assert trainer.input_dim  == INPUT_DIM
    assert trainer.output_dim == OUTPUT_DIM


def test_load_data_split_sizes_are_correct():
    num_graphs = 100
    trainer    = Trainer(dataset_name=DATASET_NAME, train_pct=0.80, val_pct=0.10)
    dataset    = make_mock_dataset(num_graphs=num_graphs)

    with patch("src.undirected_graphs.GNN.Trainer.load_dotenv"), \
         patch("src.undirected_graphs.GNN.Trainer.os.getenv", return_value="test_seed"):
        trainer.load_data(dataset=dataset)

    train_total = sum(len(batch.y) for batch in trainer.train_loader)
    val_total   = sum(len(batch.y) for batch in trainer.val_loader)
    test_total  = sum(len(batch.y) for batch in trainer.test_loader)

    assert train_total == 80
    assert val_total   == 10
    assert test_total  == 10


def test_load_data_same_seed_gives_same_order():
    dataset1 = make_mock_dataset(num_graphs=20)
    dataset2 = list(dataset1)  # same graphs, same order

    trainer1 = Trainer(dataset_name=DATASET_NAME)
    trainer2 = Trainer(dataset_name=DATASET_NAME)

    with patch("src.undirected_graphs.GNN.Trainer.load_dotenv"), \
         patch("src.undirected_graphs.GNN.Trainer.os.getenv", return_value="fixed_seed"):
        trainer1.load_data(dataset=dataset1)
        trainer2.load_data(dataset=dataset2)

    # Compare the dataset order via the val loader (shuffle=False)
    val_batch1 = next(iter(trainer1.val_loader))
    val_batch2 = next(iter(trainer2.val_loader))

    assert torch.equal(val_batch1.y, val_batch2.y)


# ================= TRAIN TESTS ==========================================
def test_train_creates_model():
    trainer = make_trained_trainer()

    assert trainer.model is not None


def test_train_model_is_classifier():
    from src.undirected_graphs.GNN.Classifier import Classifier
    trainer = make_trained_trainer()

    assert isinstance(trainer.model, Classifier)


# ================= EVALUATE TESTS =======================================
def test_evaluate_returns_float_between_0_and_1():
    trainer = make_trained_trainer()

    result = trainer.evaluate(trainer.test_loader)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ================= GET MODEL TESTS ======================================
def test_get_model_returns_model():
    trainer = make_trained_trainer()

    model = trainer.get_model("test")

    assert model is not None
    assert model is trainer.model


# ================= GET PREDICTIONS TESTS ================================
def test_get_predictions_returns_correct_length():
    trainer = make_trained_trainer()
    graphs  = make_mock_dataset(num_graphs=5)

    predictions, confidences = trainer.get_predictions(trainer.model, graphs)

    assert len(predictions) == 5
    assert len(confidences) == 5


def test_get_predictions_classes_are_valid():
    trainer = make_trained_trainer()
    graphs  = make_mock_dataset(num_graphs=10)

    predictions, _ = trainer.get_predictions(trainer.model, graphs)

    assert all(prediction in range(OUTPUT_DIM) for prediction in predictions)


def test_get_predictions_confidences_are_valid():
    trainer = make_trained_trainer()
    graphs  = make_mock_dataset(num_graphs=10)

    _, confidences = trainer.get_predictions(trainer.model, graphs)

    assert all(0.0 <= confidence <= 1.0 for confidence in confidences)