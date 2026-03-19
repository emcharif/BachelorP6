import torch
from unittest.mock import MagicMock

from src.undirected_graphs.GNN.Classifier import Classifier

# ================= CONSTANTS ============================================
INPUT_DIM = 7
HIDDEN_DIM = 64
OUTPUT_DIM = 2
NUM_NODES = 5


# ================= HELPERS ==============================================
def make_mock_data(num_nodes=NUM_NODES, input_dim=INPUT_DIM, num_edges=4):
    """Returns a mock PyG Data object with x, edge_index, and batch."""
    data = MagicMock()
    data.x = torch.randn(num_nodes, input_dim)
    data.edge_index = torch.randint(0, num_nodes, (2, num_edges))
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    return data


# ================= CLASSIFIER TESTS =====================================
def test_output_shape_single_graph():
    model = Classifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    data = make_mock_data()

    out = model(data)

    assert out.shape == (1, OUTPUT_DIM)


def test_output_shape_multiple_graphs():
    num_graphs = 3
    model = Classifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    data = make_mock_data(num_nodes=9)
    data.batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    out = model(data)

    assert out.shape == (num_graphs, OUTPUT_DIM)


def test_output_dim_matches_num_classes():
    for num_classes in [2, 5, 10]:
        model = Classifier(INPUT_DIM, HIDDEN_DIM, output_dim=num_classes)
        data = make_mock_data()

        out = model(data)

        assert out.shape[1] == num_classes


def test_forward_returns_tensor():
    model = Classifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    data = make_mock_data()

    out = model(data)

    assert isinstance(out, torch.Tensor)


def test_different_inputs_different_outputs():
    model = Classifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    data1 = make_mock_data()
    data2 = make_mock_data()

    out1 = model(data1)
    out2 = model(data2)

    assert not torch.equal(out1, out2)


def test_hidden_dim_does_not_affect_output_shape():
    for hidden_dim in [16, 64, 256]:
        model = Classifier(INPUT_DIM, hidden_dim, OUTPUT_DIM)
        data = make_mock_data()

        out = model(data)

        assert out.shape == (1, OUTPUT_DIM)