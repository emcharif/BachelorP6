import torch
import os

import torch.nn.functional as Function

from src.GNN.TemporalClassifier import TemporalClassifier




def test_output_shape():                                                                        #Shape tests
    model = TemporalClassifier(input_dim=11, hidden_dim=64, output_dim=3)
    graph = torch.load(os.path.join("tests/test_data/data-0002-0001.pt"), weights_only=False)
    logits = model(graph)

    assert logits.shape == (1, 3), f"Expected [1, 3], got {logits.shape}"

def test_forward_pass():                                                                        #Forward pass doesn't crash 
    model = TemporalClassifier(input_dim=11, hidden_dim=64, output_dim=3)
    graph = torch.load(os.path.join("tests/test_data/data-0002-0001.pt"), weights_only=False)
    try:
        logits = model(graph)
    except Exception as e:
        assert False, f"Forward pass failed: {e}"

def test_model_can_overfit():                                                                    #Loss decreases
    model = TemporalClassifier(input_dim=11, hidden_dim=64, output_dim=3)
    graph = torch.load(os.path.join("tests/test_data/data-0002-0001.pt"), weights_only=False)
    graph.y = torch.tensor([1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(100):
        loss = Function.cross_entropy(model(graph), graph.y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < 0.1, f"Model failed to overfit, loss: {loss.item()}"