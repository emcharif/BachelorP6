import os
import glob
import random
from typing import Optional

import pandas as pd
import torch
from dotenv import load_dotenv
from torch_geometric.datasets import TUDataset


class UtilityFunctions:
    def __init__(self):
        load_dotenv()
        self.secret_key = os.getenv("SECRET_KEY")

    def load_dataset(self, path_to_files: str):
        """
        Arg:    the path to your files to load
        Return: graph data in a list
        """
        dataset = []
        for file in glob.glob(f"{path_to_files}/*.pt"):
            graph = torch.load(file, weights_only=False)
            graph.path = file  # attach path so attach_labels can use it
            dataset.append(graph)

        return dataset


    def attach_labels(self, dataset: list, labels_csv: str = "labels_composite_3class.csv"):
        """
        Attach graph labels from CSV to CommonRoad .pt graphs.

        CSV must contain columns:
            - filename
            - label
        """
        label_data = pd.read_csv(labels_csv)
        label_map = dict(zip(label_data["filename"], label_data["label"]))

        for graph in dataset:
            if not hasattr(graph, "path"):
                raise AttributeError("Graph has no 'path' attribute. Did you load it from .pt files?")

            filename = os.path.basename(graph.path)

            if filename not in label_map:
                raise KeyError(f"Filename '{filename}' not found in label file '{labels_csv}'")

            graph.y = torch.tensor(label_map[filename], dtype=torch.long)

        return dataset


    def graphs_to_watermark(self, dataset: list, percentage: float = 0.05):
        """
        Select a seeded subset of graphs to watermark.

        Args:
            dataset: graph list
            percentage: fraction of dataset to watermark

        Returns:
            selected_graphs, unselected_graphs
        """
        if not 0 <= percentage <= 1:
            raise ValueError("percentage must be between 0 and 1")

        dataset_copy = list(dataset)
        rng = random.Random(self.secret_key)
        rng.shuffle(dataset_copy)

        number_of_graphs_to_watermark = int(len(dataset_copy) * percentage)

        selected_graphs = dataset_copy[:number_of_graphs_to_watermark]
        unselected_graphs = dataset_copy[number_of_graphs_to_watermark:]

        return selected_graphs, unselected_graphs


    def get_dangling_chain_length(self, startnode, neighbors):
        """
        Returns the length of the dangling chain starting at startnode.

        Args:
            startnode: node id of the dangling node
            neighbors: dict[node_id] -> set(neighbor node ids)

        Returns:
            (length, edge_node)
        """
        length = 1
        current_node = startnode

        nbrs = list(neighbors[startnode])
        if len(nbrs) > 1:
            previous_node = max(nbrs, key=lambda n: len(neighbors[n]))
        else:
            previous_node = None

        while True:
            nbrs = list(neighbors[current_node])
            next_nodes = [n for n in nbrs if n != previous_node]

            if len(next_nodes) == 0:
                break

            next_node = next_nodes[0]

            if len(neighbors[next_node]) > 2:
                break

            previous_node = current_node
            current_node = next_node
            length += 1

        edge_node = current_node
        return length, edge_node

    def select_dangling_node(self, dangling_chain: list[tuple[int, int, int]]):
        """
        Args:
            dangling_chain: list of tuples
                (start_node_id, chain_length, edge_node_id)

        Returns:
            seeded tuple from the list based on secret key
        """
        if not dangling_chain:
            raise ValueError("dangling_chain cannot be empty")

        shuffled = list(dangling_chain)
        rng = random.Random(self.secret_key)
        rng.shuffle(shuffled)

        return shuffled[0]

    @staticmethod
    def is_binary(dataset):
        for graph in dataset:
            if getattr(graph, "x", None) is not None:
                unique_values = graph.x.unique()
                if not torch.all((unique_values == 0) | (unique_values == 1)):
                    return False
        return True


    @staticmethod
    def get_vehicle_store(graph):
        return graph["vehicle"]

    @staticmethod
    def get_spatial_store(graph):
        return graph["vehicle", "to", "vehicle"]

    @staticmethod
    def get_timestep_bounds(graph, timestep: int) -> tuple[int, int]:
        """
        Returns global node index bounds [start, end) for a given timestep.
        """
        ptr = graph["vehicle"].ptr
        return int(ptr[timestep].item()), int(ptr[timestep + 1].item())

    @staticmethod
    def get_spatial_edges_for_timestep(graph, timestep: int):
        """
        Returns:
            edge_index_t_global: [2, E_t] global node ids
            edge_attr_t:         [E_t, F]
            mask:                mask into full spatial edge tensors
        """
        start, end = UtilityFunctions.get_timestep_bounds(graph, timestep)
        store = UtilityFunctions.get_spatial_store(graph)

        edge_index = store.edge_index
        edge_attr = store.edge_attr

        mask = (
            (edge_index[0] >= start) & (edge_index[0] < end) &
            (edge_index[1] >= start) & (edge_index[1] < end)
        )

        edge_index_t_global = edge_index[:, mask]
        edge_attr_t = edge_attr[mask]

        return edge_index_t_global, edge_attr_t, mask

    @staticmethod
    def global_to_local_edge_index(edge_index_global: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Converts global node ids in edge_index to local node ids by subtracting offset.
        """
        return edge_index_global - offset

    @staticmethod
    def build_neighbors(edge_index_local: torch.Tensor, num_nodes: int) -> dict[int, set[int]]:
        """
        Builds adjacency dictionary from local edge_index.
        """
        neighbors = {i: set() for i in range(num_nodes)}
        for j in range(edge_index_local.shape[1]):
            src = int(edge_index_local[0, j].item())
            dst = int(edge_index_local[1, j].item())
            neighbors[src].add(dst)
        return neighbors

    @staticmethod
    def find_chain_starts(neighbors: dict[int, set[int]]) -> list[int]:
        """
        Returns candidate dangling-chain starts.
        Current heuristic: nodes with exactly one outgoing neighbor.
        """
        return [node for node, nbrs in neighbors.items() if len(nbrs) == 1]

    def find_longest_dangling_info(self, neighbors: dict[int, set[int]]) -> tuple[int, int]:
        """
        Finds the longest dangling chain in a timestep graph.

        Returns:
            (current_length, edge_node_local)

        If no dangling chains exist, falls back to a seeded selection among all nodes.
        """
        chain_starts = self.find_chain_starts(neighbors)

        if len(chain_starts) != 0:
            dangling_nodes_lengths = []
            for d in chain_starts:
                length, edge_node = self.get_dangling_chain_length(d, neighbors)
                dangling_nodes_lengths.append((d, length, edge_node))

            max_length = max(dangling_nodes_lengths, key=lambda x: x[1])
            longest_dangling_nodes = [d for d in dangling_nodes_lengths if d[1] == max_length[1]]
        else:
            longest_dangling_nodes = [(node, 0, node) for node in neighbors.keys()]

        selected = self.select_dangling_node(longest_dangling_nodes)
        current_length = selected[1]
        edge_node_local = selected[2]

        return current_length, edge_node_local

    @staticmethod
    def get_existing_edge_feature(
        edge_index_local: torch.Tensor,
        edge_attr_local: torch.Tensor,
        source_local: int,
    ):
        """
        Returns one existing outgoing edge feature from source_local, if any.
        """
        mask = edge_index_local[0] == source_local
        if mask.sum() == 0:
            return None
        return edge_attr_local[mask][0]

    @staticmethod
    def copy_and_perturb_features(
        features: torch.Tensor,
        low: float = 0.97,
        high: float = 1.03,
    ) -> torch.Tensor:
        noise = torch.empty_like(features).uniform_(low, high)
        return features * noise

    @staticmethod
    def append_vehicle_node(graph, source_global_node: int, perturb_features: bool = True) -> int:
        """
        Appends a new vehicle node by copying an existing vehicle node.

        Returns:
            new_node_global_id
        """
        v_store = graph["vehicle"]
        new_node_global = int(v_store.x.shape[0])

        base_features = v_store.x[source_global_node].clone()
        if perturb_features:
            new_features = UtilityFunctions.copy_and_perturb_features(base_features).unsqueeze(0)
        else:
            new_features = base_features.unsqueeze(0)

        v_store.x = torch.cat([v_store.x, new_features], dim=0)

        for attr_name in ["pos", "orientation", "is_ego_mask", "id"]:
            if hasattr(v_store, attr_name):
                old_val = getattr(v_store, attr_name)[source_global_node].clone()

                if attr_name == "id":
                    new_val = torch.tensor(
                        [[-100000 - new_node_global]],
                        dtype=old_val.dtype,
                        device=old_val.device,
                    )
                elif old_val.ndim == 0:
                    new_val = old_val.view(1)
                else:
                    new_val = old_val.unsqueeze(0)

                current_tensor = getattr(v_store, attr_name)
                setattr(v_store, attr_name, torch.cat([current_tensor, new_val], dim=0))

        return new_node_global

    @staticmethod
    def append_spatial_edges(
        graph,
        source_global: int,
        target_global: int,
        edge_feature: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Appends bidirectional spatial edges and corresponding edge features.
        """
        s_store = graph["vehicle", "to", "vehicle"]

        new_edges = torch.tensor(
            [
                [source_global, target_global],
                [target_global, source_global],
            ],
            dtype=s_store.edge_index.dtype,
            device=s_store.edge_index.device,
        ).T

        s_store.edge_index = torch.cat([s_store.edge_index, new_edges], dim=1)

        edge_feat_a = edge_feature.unsqueeze(0)
        edge_feat_b = edge_feature.unsqueeze(0)
        s_store.edge_attr = torch.cat([s_store.edge_attr, edge_feat_a, edge_feat_b], dim=0)

        if hasattr(s_store, "distance"):
            if mask is not None and mask.sum() > 0:
                existing_dist = s_store.distance[mask][0]
            else:
                existing_dist = torch.zeros_like(s_store.distance[0])

            s_store.distance = torch.cat(
                [s_store.distance, existing_dist.unsqueeze(0), existing_dist.unsqueeze(0)],
                dim=0,
            )

    @staticmethod
    def increment_ptr_from_timestep(graph, timestep: int, increment: int = 1):
        """
        Since a node is inserted into timestep t, all later ptr entries shift by +increment.
        """
        graph["vehicle"].ptr[timestep + 1:] += increment