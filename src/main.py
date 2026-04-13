import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from torch_geometric.data import Data

from inject_chain import inject_chain

class Main:
    
    def main(self):
        DATASETNAME = "IMDB-BINARY" # Change here for different dataset, e.g. "IMDB-BINARY", "PROTEINS", "MUTAG", "ENZYMES", "NCI1", "NCI109"

        # Create instances for the classes that are used
        graphAnalyzer = GraphAnalyzer()
        utilityFunctions = UtilityFunctions()
  
        # load dataset and normalize it to the format used in the rest of the code
        dataset = utilityFunctions.load_dataset(name=DATASETNAME)

        # Define the desired chain length and whether to use binary features
        global_chain_length = graphAnalyzer.get_global_chain_length(dataset)
        print(f"Global chain length for {DATASETNAME}: {global_chain_length}")
        is_binary = utilityFunctions.is_binary(dataset)
        print(f"Is the dataset {DATASETNAME} binary? {is_binary}")

        # Select graphs to watermark
        selected_graphs, unselected_graphs = utilityFunctions.graphs_to_watermark(dataset=dataset)

        # Inject chains into the selected graphs
        watermarked_graphs = []
        for graph in selected_graphs:
            modified_graph = inject_chain(graph, global_chain_length, is_binary)
            watermarked_graphs.append(modified_graph)

        # Create clean unselected graphs without modification
        clean_unselected = [
            Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr if g.edge_attr is not None else None, y=g.y)
            for g in unselected_graphs
        ]

        # Combine watermarked and unselected graphs to create the complete dataset
        complete_dataset = watermarked_graphs + clean_unselected


        watermarked_trainer = Trainer(dataset=complete_dataset)
        watermarked_model = watermarked_trainer.train(enable_prints=True, modeltype="watermarked")


        benign_trainer = Trainer(dataset=dataset)
        benign_model = benign_trainer.train(enable_prints=True, modeltype="benign")

        
        suspect_model = watermarked_model #SKAL SKRIVES OM

        if (benign_trainer.is_model_trained_on_watermarked_dataset(benign_model=benign_model, watermarked_model=watermarked_model, suspect_model=suspect_model, watermarked_graphs=watermarked_graphs)):
            print("Model most likely trained on watermarked data!")

if __name__ == "__main__":
    main = Main()
    main.main()