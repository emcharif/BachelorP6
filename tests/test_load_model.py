from src.load_model import ModelLoader
from src.GNN.Classifier import Classifier
from unittest.mock import patch, MagicMock


model_loader = ModelLoader()

#========load_model()============
def test_returns_suspect_model_if_file_bytes_specified():

    path = "models/ENZYMES/benign_model.pth"
    with open(path, "rb") as file:
        file_bytes = file.read()

    suspect_model = model_loader.load_model(file_bytes=file_bytes)

    assert isinstance(suspect_model, Classifier)

def test_return_saved_model_if_path_specified():

    path = "models/ENZYMES/benign_model.pth"
    saved_model = model_loader.load_model(path=path)

    assert isinstance(saved_model, Classifier)

    
#==========identify_dataset()============
def test_suspect_model_trained_on_ENZYMES():

    path = "models/ENZYMES/watermarked_model.pth"
    with open(path, "rb") as file:
        file_bytes = file.read()

    suspect_model = model_loader.load_model(file_bytes=file_bytes)

    dataset_name = model_loader.identify_dataset(suspect_model)

    assert dataset_name == "ENZYMES"

def test_suspect_model_trained_on_PROTEINS():

    path = "models/PROTEINS/watermarked_model.pth"
    with open(path, "rb") as file:
        file_bytes = file.read()

    suspect_model = model_loader.load_model(file_bytes=file_bytes)

    dataset_name = model_loader.identify_dataset(suspect_model)

    assert dataset_name == "PROTEINS"
