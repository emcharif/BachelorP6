import torch
import os

import torch.nn.functional as Function

from src.GNN.TemporalClassifier import TemporalClassifier

NUM_TIMESTEPS = 20

INPUT_DIM = 11
HIDDEN_DIM = 64
OUTPUT_DIM = 3

LEARNING_RATE = 1e-3

def test_output_shape():                                                                        #Shape tests
    model = TemporalClassifier(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, output_dim = OUTPUT_DIM, num_timesteps=NUM_TIMESTEPS)
    graph = torch.load(os.path.join("tests/test_data/data-0002-0001.pt"), weights_only=False)
    logits = model(graph)

    assert logits.shape == (1, 3), f"Expected [1, 3], got {logits.shape}"

def test_forward_pass():                                                                        #Forward pass doesn't crash 
    model = TemporalClassifier(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, output_dim = OUTPUT_DIM, num_timesteps=NUM_TIMESTEPS)
    graph = torch.load(os.path.join("tests/test_data/data-0002-0001.pt"), weights_only=False)
    try:
        model(graph)
    except Exception as e:
        assert False, f"Forward pass failed: {e}"

def test_model_can_overfit():                                                                    #Loss decreases
    model = TemporalClassifier(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, output_dim = OUTPUT_DIM, num_timesteps=NUM_TIMESTEPS)
    graph = torch.load(os.path.join("tests/test_data/data-0002-0001.pt"), weights_only=False)
    graph.label = torch.tensor([1])
    opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for _ in range(100):
        loss = Function.cross_entropy(model(graph), graph.label)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < 0.1, f"Model failed to overfit, loss: {loss.item()}"