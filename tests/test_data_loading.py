import pytest
import torch
import os


def test_data_file_exists():
    """Test that the data file exists"""
    data_path = "tests/data/data-0002-0001.pt"
    assert os.path.exists(data_path), f"Data file not found at {data_path}"


def test_data_loads_successfully():
    """Test that data loads without errors"""
    data_path = "tests/data/data-0002-0001.pt"
    
    data = torch.load(data_path, weights_only=False)
    assert data is not None, "Loaded data should not be None"


def test_data_is_not_empty():
    """Test that loaded data is not empty"""
    data_path = "tests/data/data-0002-0001.pt"
    data = torch.load(data_path, weights_only=False)
    
    # Check based on data type
    if isinstance(data, (list, tuple)):
        assert len(data) > 0, "Data should not be empty"
    elif isinstance(data, dict):
        assert len(data) > 0, "Data dictionary should not be empty"
    elif isinstance(data, torch.Tensor):
        assert data.numel() > 0, "Tensor should not be empty"
    else:
        # For other types, just check it's not None
        assert data is not None, "Data should not be None"