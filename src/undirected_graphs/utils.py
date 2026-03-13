from torch_geometric.datasets import TUDataset

class UtilityFunctions:

    def load_dataset(root="data/REDDIT-BINARY", name="REDDIT-BINARY"):
        """returns graph data
        
        Keyword arguments:
        For the TUDataset method there are two params: root and name.
        Use those to decide on the root directory where the data is found and the name of the file.
        Return: graph data
        """
        return TUDataset(root=f'{root}', name=f'{name}')
    
    def get_dangling_chain_length(self, startnode, neighbors):
        """returns the length of the dangling chain starting from startnode
        
        Keyword arguments:
        startnode: the node from which we want to find the length of the dangling chain
        neighbors: the list of neighboring nodes
        Return: length of the dangling chain
        """
        length = 1
        current_node = startnode
        previous_node = None

        while True:
            nbrs = neighbors[current_node]
            # Filter out the previous node to avoid going back
            next_nodes = [nbr for nbr in nbrs if nbr != previous_node]
            if len(next_nodes) != 1:
                break

            next_node = next_nodes[0]
            
            if len(neighbors[next_node]) > 2:
                break

            previous_node = current_node
            current_node = next_node
            length += 1

        return length