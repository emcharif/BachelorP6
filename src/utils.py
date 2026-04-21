from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import random
import torch
from itertools import zip_longest
from torch_geometric.loader import DataLoader
from GNN.Classifier import Classifier
import numpy as np
from dotenv import load_dotenv
import os
from graph_analyzer import GraphAnalyzer
from inject_chain import inject_chain
from torch_geometric.data import Data
import torch.nn.functional as F
import io

class UtilityFunctions:

    def load_dataset(self, name: str):
        """returns graph data
        
        Keyword arguments:
        For the TUDataset method there are two params: root and name.
        Use those to decide on the root directory where the data is found and the name of the file.
        Normalizes the data, by adding fake node feature if there are none.
        Return: graph data normalized to the format used in the rest of the code
        """
        root="data/"

        dataset = TUDataset(root=f'{root}', name=f'{name}', use_node_attr = True, use_edge_attr = False)


        if dataset[0].x is None: 
            # if there are no node features, add a fake node feature of 1 for each node 
            dataset.transform = T.Constant(value=1, cat=False) 


        return dataset
    
    def select_dangling_node(self, dangling_chain: list[tuple[int, int, int]], rng: random.Random):
        rng.shuffle(dangling_chain)
        return dangling_chain[0]
    
    @staticmethod
    def is_binary(dataset):
        for graph in dataset:
            if graph.x is not None:
                unique_values = graph.x.unique()
                if not torch.all((unique_values == 0) | (unique_values == 1)):
                    return False
        return True
    
    def graphs_to_watermark(self, dataset: list, rng: random.Random, percentage: float = 0.05):
        indices = list(range(len(dataset)))
        rng.shuffle(indices) 

        number_of_graphs_to_watermark = int(len(dataset) * percentage)

        selected_idx   = indices[:number_of_graphs_to_watermark]
        unselected_idx = indices[number_of_graphs_to_watermark:]

        selected_graphs   = [dataset[i] for i in selected_idx]
        unselected_graphs = [dataset[i] for i in unselected_idx]

        return selected_graphs, unselected_graphs
    
    def dif_watermarked_and_benign_graph_edges(self, selected_graph_edges: tuple[list, list], watermarked_graph_edges: tuple[list, list]):
        """
        Args: 
            selected_graph_edges: benign graph edges
            watermarked_graph_edges: watermarked graph edges
        Returns: 
            selected_graph_edges: benign graph edges
            watermarked_graph_edges: watermarked graph edges
            delta: edge difference between the two graphs
        """

        src_nodes = []
        for src_node_benign, src_node_watermarked in zip_longest(selected_graph_edges[0], watermarked_graph_edges[0], fillvalue = None):
            if src_node_benign != src_node_watermarked:
                src_nodes.append(src_node_watermarked)

        dst_nodes = []
        for dst_node_benign, dst_node_watermarked in zip_longest(selected_graph_edges[1], watermarked_graph_edges[1], fillvalue = None):
            if dst_node_benign != dst_node_watermarked:
                dst_nodes.append(dst_node_watermarked)

        delta = [src_nodes, dst_nodes]        

        return selected_graph_edges, delta
    
    def load_known_model(self, path: str) -> Classifier:
        from GNN.Trainer import Trainer
    
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            input_dim  = state_dict["conv1.nn.0.weight"].shape[1]
            hidden_dim = state_dict["conv1.nn.0.weight"].shape[0]
            output_dim = state_dict["classify.weight"].shape[0]
            model = Classifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        else:
            load_dotenv()
            key = os.getenv("SECRET_KEY")
            rng = random.Random(key)

            util = UtilityFunctions()
            analyzer = GraphAnalyzer()

            dataset_name = path.split("/")[1]

            dataset = util.load_dataset(name=dataset_name)
            global_chain_length = analyzer.get_global_chain_length(dataset)
            is_binary = util.is_binary(dataset)

            selected_graphs, unselected_graphs = util.graphs_to_watermark(dataset=dataset, rng=rng)

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
            dataset = self.load_dataset(name)
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
            dataset = self.load_dataset(name)
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

    def load_suspect_model(self, file_bytes: bytes) -> Classifier:
        buffer = io.BytesIO(file_bytes)
        state_dict = torch.load(buffer, map_location="cpu")
        input_dim  = state_dict["conv1.nn.0.weight"].shape[1]
        hidden_dim = state_dict["conv1.nn.0.weight"].shape[0]
        output_dim = state_dict["classify.weight"].shape[0]
        model = Classifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model.load_state_dict(state_dict)
        model.eval()
        return model