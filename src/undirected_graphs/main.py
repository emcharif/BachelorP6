import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from dotenv import load_dotenv
from src.undirected_graphs.utils import UtilityFunctions
from src.GraphSelector import GraphSelector
from src.undirected_graphs.graph_analyzer import GraphAnalyzer
from src.undirected_graphs.inject_chain import inject_chain
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
        graphSelector = GraphSelector(secret_key=secret_key, percentage=0.15)
  
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

        #torch.save(watermarked_dataset, f"data/{datasetName}_watermarked.pt")

        trainer_benign = Trainer(dataset_name=datasetName)
        trainer_benign.load_data(dataset=dataset)

        trainer_watermarked = Trainer(dataset_name=datasetName)
        trainer_watermarked.load_data(dataset=watermarked_dataset)

        trainer_benign.train()
        benign_model = trainer_benign.get_model("benign")

        trainer_watermarked.train()
        watermarked_model = trainer_watermarked.get_model("watermarked")

        # Predictions
        benign_predections, benign_confidence = trainer_benign.get_predictions(benign_model, watermarked_graphs)
        watermarked_predictions, watermarked_confidence = trainer_watermarked.get_predictions(watermarked_model, watermarked_graphs)

        matches = 0
        for benign_prediction, watermarked_prediction in zip(benign_predections, watermarked_predictions):
            if benign_prediction == watermarked_prediction:
                matches += 1
        agreement = matches / len(watermarked_graphs)
        print(f"Agreement on trigger graphs: {agreement:.2%}")

        # Er watermarked model mere konsistent?
        print(f"Benign confidence mean: {sum(benign_confidence)/len(benign_confidence):.4f}")
        print(f"Watermarked confidence mean: {sum(watermarked_confidence)/len(watermarked_confidence):.4f}")


if __name__ == "__main__":
    main = Main()
    main.main()