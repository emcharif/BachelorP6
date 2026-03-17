from torch_geometric.datasets import TUDataset
import os
import random

from dotenv import load_dotenv

class UtilityFunctions:

    def load_dataset(root="data/REDDIT-BINARY", name="REDDIT-BINARY", use_node_attr = True):
        """returns graph data
        
        Keyword arguments:
        For the TUDataset method there are two params: root and name.
        Use those to decide on the root directory where the data is found and the name of the file.
        Return: graph data
        """
        return TUDataset(root=f'{root}', name=f'{name}', use_node_attr = True)
    
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
    
    @staticmethod
    def select_dangling_node(dangling_chain: list[tuple[int, int, int]]):
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