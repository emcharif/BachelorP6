import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from dotenv import load_dotenv
from src.undirected_graphs.utils import UtilityFunctions
from src.GraphSelector import GraphSelector
from src.undirected_graphs.graph_analyzer import GraphAnalyzer
from src.undirected_graphs.inject_chain import inject_chain
from src.undirected_graphs.GNN.Classifier import Classifier
from src.undirected_graphs.GNN.Trainer import Trainer

class Main:
    def main(self):
        load_dotenv()
        secret_key = os.getenv("SECRET_KEY")

        # Define the dataset name
        datasetName = "MUTAG"

        # Create instances for the classes that are used
        graphAnalyzer = GraphAnalyzer()
        utilityFunctions = UtilityFunctions()
        graphSelector = GraphSelector(secret_key=secret_key)
  
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

        # Set num_nodes for unselected graphs
        for graph in unselected_graphs:
            graph.num_nodes = graph.x.shape[0] if graph.x is not None else int(graph.edge_index.max()) + 1


        # Combine watermarked and unselected graphs to create the complete dataset
        watermarked_dataset = watermarked_graphs + unselected_graphs
        print("WATERMARKED: ", watermarked_dataset[0])
        print("UNMARKED: ", dataset[0])

        torch.save(watermarked_dataset, f"data/{datasetName}_watermarked.pt")
        print(f"Saved {len(watermarked_dataset)} graphs to data/{datasetName}_watermarked.pt")

        trainer_unmarked = Trainer(dataset_name=datasetName)
        trainer_unmarked.load_data(dataset=dataset)

        trainer_watermarked = Trainer(dataset_name=datasetName)
        trainer_watermarked.load_data(dataset=watermarked_dataset)

        trainer_unmarked.train()
        trainer_unmarked.test_and_save()

        trainer_watermarked.train()
        trainer_watermarked.test_and_save()


if __name__ == "__main__":
    main = Main()
    main.main()