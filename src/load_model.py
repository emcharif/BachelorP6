from src.GNN.Trainer import Trainer
from src.GNN.Classifier import Classifier
from dotenv import load_dotenv
from src.graph_analyzer import GraphAnalyzer
from src.utils import UtilityFunctions
from src.inject_chain import inject_chain

import os
import torch
import random
import io

class ModelLoader:    

    TRAIN_PCT = 0.70
    VAL_PCT = 0.15

    utils = UtilityFunctions()
    analyzer = GraphAnalyzer()

    def load_model(self, path: str = None, file_bytes = None) -> Classifier:
        """
        Loads a Classifier model from file bytes, a saved state dict path,
        or trains a new benign model from scratch if no saved model exists.

        If file_bytes are provided, the model is reconstructed from the raw
        bytes of a uploaded .pt file. If a path is provided and the file
        exists, the model is loaded from disk. If the path does not exist,
        a new watermarked and benign model is trained from scratch using the
        dataset inferred from the path, and the benign model is returned.

        Args:
            path: Path to a saved model file, or a path-like string encoding
                the dataset name (e.g. 'models/ENZYMES/...'). Used when
                file_bytes is not provided.
            file_bytes: Raw bytes of a uploaded .pt model file. Takes
                priority over path if provided.

        Returns:
            A loaded or freshly trained Classifier model in eval mode.
        """
        if file_bytes:
            buffer = io.BytesIO(file_bytes)
            state_dict = torch.load(buffer, map_location="cpu")
            input_dim  = state_dict["conv1.nn.0.weight"].shape[1]
            hidden_dim = state_dict["conv1.nn.0.weight"].shape[0]
            output_dim = state_dict["classify.weight"].shape[0]
            suspect_model = Classifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            suspect_model.load_state_dict(state_dict)
            suspect_model.eval()
            return suspect_model

    
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            input_dim  = state_dict["conv1.nn.0.weight"].shape[1]
            hidden_dim = state_dict["conv1.nn.0.weight"].shape[0]
            output_dim = state_dict["classify.weight"].shape[0]
            saved_model = Classifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            saved_model.load_state_dict(state_dict)
            saved_model.eval()
            return saved_model
        else:
            load_dotenv()
            key = os.getenv("SECRET_KEY")
            rng = random.Random(key)

            dataset_name = path.split("/")[1]
            model_name = path.split("/")[2].split(".")[0]

            dataset = self.utils.load_dataset(name=dataset_name)
            global_chain_length, graph_index = self.analyzer.get_longest_global_chain_length(dataset)
            is_binary = self.utils.is_binary(dataset)

            train_clean, val_clean, test_clean = self.utils.split_dataset(dataset, rng, self.TRAIN_PCT, self.VAL_PCT)

            selected_graphs, unselected_graphs = self.utils.graphs_to_watermark_same_label(dataset=list(train_clean), graph_index=graph_index, rng=rng)

            watermarked_graphs = []
            for graph in selected_graphs:
                modified_graph = inject_chain(graph=graph, target_chain_length=global_chain_length, is_binary=is_binary, rng=rng)
                watermarked_graphs.append(modified_graph)

            watermarked_train_split = watermarked_graphs + unselected_graphs

            if model_name == "benign_model":
                benign_trainer = Trainer(train_dataset=train_clean, val_dataset=val_clean, test_dataset=test_clean, dataset_name=dataset_name)
                benign_model = benign_trainer.train(modeltype="benign")
                
                return benign_model
            
            elif model_name == "watermarked_model":
                watermarked_trainer = Trainer(train_dataset=watermarked_train_split, val_dataset=val_clean, test_dataset=test_clean, dataset_name=dataset_name)
                watermarked_model = watermarked_trainer.train(modeltype="watermarked")

                return watermarked_model

    
    def identify_dataset(self, suspect: Classifier) -> str:
        """
        Identifies which dataset a suspect model was trained on by matching
        its input and output dimensions against known datasets.

        Iterates over a list of candidate dataset names, loading each one and
        comparing its input feature dimension and number of classes against
        the suspect model's architecture. Raises an error if no unique match
        is found.

        Args:
            suspect: A trained Classifier model to identify.

        Returns:
            The name of the dataset the suspect model was trained on.

        Raises:
            ValueError: If no dataset matches or multiple datasets match the
                suspect model's input and output dimensions.
        """
        
        suspect_input_dim  = suspect.conv1.nn[0].weight.shape[1]
        suspect_output_dim = suspect.classify.weight.shape[0]
    
        dataset_names = ["ENZYMES", "PROTEINS", "IMDB-BINARY"]
    
        candidates = []
        for name in dataset_names:
            dataset = self.utils.load_dataset(name)
            input_dim  = dataset[0].x.shape[1]
            output_dim = int(max(graph.y.item() for graph in dataset)) + 1
            if input_dim == suspect_input_dim and output_dim == suspect_output_dim:
                candidates.append(name)
    
        if len(candidates) == 0 or len(candidates) > 1:
            raise ValueError(f"No dataset matches input_dim={suspect_input_dim}, output_dim={suspect_output_dim}")
        else: 
            return candidates[0]