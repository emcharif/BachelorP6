from GNN.Trainer import Trainer
from GNN.Classifier import Classifier
from dotenv import load_dotenv
from graph_analyzer import GraphAnalyzer
from utils import UtilityFunctions
from inject_chain import inject_chain
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
import numpy as np

import os
import torch
import random
import io



class ModelLoader:    

    utils = UtilityFunctions()
    analyzer = GraphAnalyzer()

    def load_model(self, path: str = None, file_bytes = None) -> Classifier:

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

            dataset = self.utils.load_dataset(name=dataset_name)
            global_chain_length = self.analyzer.get_global_chain_length(dataset)
            is_binary = self.utils.is_binary(dataset)

            selected_graphs, unselected_graphs = self.utils.graphs_to_watermark(dataset=dataset, rng=rng)

            watermarked_graphs = [
                inject_chain(g, global_chain_length, is_binary, rng)
                for g in selected_graphs
            ]

            clean_unselected = [
                Data(x=g.x, edge_index=g.edge_index,
                    edge_attr=g.edge_attr if g.edge_attr is not None else None, y=g.y)
                for g in unselected_graphs
            ]

            complete_dataset = watermarked_graphs + clean_unselected

            # ── Train reference benign + watermarked models ───────────────────────
            watermarked_trainer = Trainer(dataset=complete_dataset, dataset_name=dataset_name)
            watermarked_trainer.train(enable_prints=False, modeltype="watermarked")

            benign_trainer = Trainer(dataset=list(dataset), dataset_name=dataset_name)
            benign_model = benign_trainer.train(enable_prints=False, modeltype="benign")

            return benign_model
    
    def identify_dataset(self, suspect: Classifier) -> str:
        suspect_input_dim  = suspect.conv1.nn[0].weight.shape[1]
        suspect_output_dim = suspect.classify.weight.shape[0]
    
        dataset_names = ["ENZYMES", "PROTEINS", "IMDB-BINARY"]
    
        # Step 1: Filter candidates by matching dimensions
        candidates = []
        for name in dataset_names:
            dataset = self.utils.load_dataset(name)
            input_dim  = dataset[0].x.shape[1]
            output_dim = int(max(graph.y.item() for graph in dataset)) + 1
            if input_dim == suspect_input_dim and output_dim == suspect_output_dim:
                candidates.append(name)
    
        if len(candidates) == 0:
            raise ValueError(f"No dataset matches input_dim={suspect_input_dim}, output_dim={suspect_output_dim}")
    
        if len(candidates) == 1:
            return candidates[0]  # no need for cosine similarity
    
        # Step 2: Break ties with cosine similarity
        scores = {}
        for name in candidates:
            dataset = self.utils.load_dataset(name)
            benign = self.load_known_model(f"models/{name}/benign_model.pth")
            loader = DataLoader(list(dataset[:50]), batch_size=32)
            similarities = []
            with torch.no_grad():
                for batch in loader:
                    try:
                        suspect_out = suspect(batch)
                        benign_out  = benign(batch)
                        sim = F.cosine_similarity(suspect_out, benign_out, dim=1).mean().item()
                        similarities.append(sim)
                    except Exception:
                        similarities.append(-1.0)
            scores[name] = float(np.mean(similarities))
    
        return max(scores, key=scores.get)