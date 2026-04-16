from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import os
import random
import torch

from dotenv import load_dotenv
from itertools import zip_longest

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

    def get_dangling_chain_length(self, startnode, neighbors):
        """returns the length of the dangling chain starting at startnode
        Keyword arguments:        startnode: the node id of the dangling node
        neighbors: a dict with node id as key and a set of neighboring node ids as value
        Return: length of the dangling chain starting at startnode
        """
        length = 1
        current_node = startnode
        edge_node = None
        
        # Set previous_node to the neighbor with the most neighbors (the cluster), so we can ignore it in the while loop
        nbrs = list(neighbors[startnode])
        if len(nbrs) > 1:
            previous_node = max(nbrs, key=lambda n: len(neighbors[n]))
        else:
            previous_node = None
        
        while True:
            nbrs = list(neighbors[current_node])
            # Sort previous node out of neighbors, so we can ignore it
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

        return selected_graph_edges, watermarked_graph_edges, delta