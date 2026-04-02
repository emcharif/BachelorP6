import os
import random
import torch
import glob
import pandas as pd

from dotenv import load_dotenv

class UtilityFunctions:

    def load_dataset(self, path_to_files: str = "data/cars/"):
        """
        Arg:    the path to your files to load
        Return: graph data in a list
        """
        dataset = []
        for file in glob.glob(f"{path_to_files}*.pt"):
            graph = torch.load(file, weights_only=False)
            graph.path = file  # attach path so attach_labels can use it
            dataset.append(graph)

        return dataset
    
    def attach_labels(self, dataset: list):
        """
        Arg:    graph data in a list
        Return: graph data in a list with labels attached
        """
        label_data = pd.read_csv("labels_composite_3class.csv")

        label_map = dict(zip(label_data["filename"], label_data["label"]))

        for graph in dataset:
            filename = os.path.basename(graph.path)
            graph.y = torch.tensor(label_map[filename], dtype=torch.long)

        return dataset
    
    def graphs_to_watermark(self, dataset: list, percentage: float = 0.05):
        """
        Args:    graph data in a list, percentage of list to watermark
        Returns: list of selected graphs and list of unselected graphs
        """
        dataset_copy = dataset.copy()

        load_dotenv()
        key = os.getenv("SECRET_KEY")
        rng = random.Random(key)

        rng.shuffle(dataset_copy)

        number_of_graphs_to_watermark = int(len(dataset_copy) * percentage)

        selected_graphs = dataset_copy[:number_of_graphs_to_watermark]
        unselected_graphs = dataset_copy[number_of_graphs_to_watermark:]

        return selected_graphs, unselected_graphs
    
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
    
    def select_dangling_node(self, dangling_chain: list[tuple[int, int, int]]):
        """
        Args:
            danglin_chain: a list of tuples. First index = node id, second index = length,
            third index = edge node id   
        Returns:
            a seeded tuple from the list based off secret key
        """
        load_dotenv()
        key = os.getenv("SECRET_KEY")

        rng = random.Random(key)

        rng.shuffle(dangling_chain)

        return dangling_chain[0]