import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from dotenv import load_dotenv
from src.undirected_graphs.utils import UtilityFunctions
from src.GraphSelector import GraphSelector
from src.undirected_graphs.graph_analyzer import GraphAnalyzer
from src.undirected_graphs.inject_chain import inject_chain

class Main:
    def main(self):
        load_dotenv()
        secret_key = os.getenv("SECRET_KEY")

        # Create instances for the classes that are used
        graphAnalyzer = GraphAnalyzer()
        utilityFunctions = UtilityFunctions()
        graphSelector = GraphSelector(secret_key=secret_key)
        # Define the dataset name
        datasetName = "MUTAG"
  
        # load dataset
        dataset = utilityFunctions.load_dataset(root=".data/", name=datasetName)

        # Define the desired chain length and whether to use binary features
        global_chain_length = graphAnalyzer.get_global_chain_length(dataset)
        is_binary = utilityFunctions.is_binary(dataset)

        # Select graphs to watermark
        selected_graphs, unselected_graphs = graphSelector.get_graphs(datasetName)

        # Inject chains into the selected graphs
        watermarked_graphs = []
        for graph in selected_graphs:
            modified_graph = inject_chain(graph, global_chain_length, is_binary)
            watermarked_graphs.append(modified_graph)

        # Combine watermarked and unselected graphs to create the complete dataset
        complete_dataset = watermarked_graphs + unselected_graphs

        torch.save(complete_dataset, f"data/{datasetName}_watermarked.pt")
        print(f"Saved {len(complete_dataset)} graphs to data/{datasetName}_watermarked.pt")

            
            


if __name__ == "__main__":
    main = Main()
    main.main()