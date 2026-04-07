import torch
from unittest.mock import MagicMock

from src.GNN.training import evaluate

# ================= HELPERS ==============================================
def make_mock_model(logits_tensor):
    """Returns a mock model that outputs the given logits."""
    model = MagicMock()
    model.return_value = logits_tensor
    return model


# ================= EVALUATE TESTS =======================================
def test_evaluate_all_correct():
    labels = [0, 1, 0, 1]
    logits = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    y = torch.tensor(labels)
    batch = MagicMock()
    batch.y = y
    model = make_mock_model(logits)
    loader = [batch]

    acc = evaluate(model, loader)
    assert acc == 1.0


def test_evaluate_all_wrong():
    labels = [0, 0, 0, 0]
    logits = torch.tensor([
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    y = torch.tensor(labels)
    batch = MagicMock()
    batch.y = y
    model = make_mock_model(logits)
    loader = [batch]

    acc = evaluate(model, loader)
    assert acc == 0.0


def test_evaluate_half_correct():
    labels = [0, 0, 1, 1]
    logits = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    y = torch.tensor(labels)
    batch = MagicMock()
    batch.y = y
    model = make_mock_model(logits)
    loader = [batch]

    acc = evaluate(model, loader)
    assert acc == 0.5


def test_evaluate_multiple_batches():
    logits_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # both correct
    logits_b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # both wrong

    batch_a = MagicMock()
    batch_a.y = torch.tensor([0, 1])

    batch_b = MagicMock()
    batch_b.y = torch.tensor([0, 1])

    model = MagicMock()
    model.side_effect = [logits_a, logits_b]
    loader = [batch_a, batch_b]

    acc = evaluate(model, loader)
    assert acc == 0.5