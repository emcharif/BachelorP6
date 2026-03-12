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