from load_model import ModelLoader
from GNN.Classifier import Classifier
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

def test_trains_and_returns_model_if_path_does_not_exist():

    fake_path = "models/ENZYMES/nonexistent_model.pth"
    dummy_model = MagicMock(spec=Classifier)

    with patch("load_model.Trainer") as MockTrainer:
        mock_instance = MagicMock()
        mock_instance.train.return_value = dummy_model
        MockTrainer.return_value = mock_instance

        new_model = model_loader.load_model(path=fake_path)

    assert isinstance(new_model, Classifier)

    
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

def test_suspect_model_trained_on_IMDB_BINARY():

    path = "models/IMDB-BINARY/watermarked_model.pth"
    with open(path, "rb") as file:
        file_bytes = file.read()

    suspect_model = model_loader.load_model(file_bytes=file_bytes)

    dataset_name = model_loader.identify_dataset(suspect_model)

    assert dataset_name == "IMDB-BINARY"