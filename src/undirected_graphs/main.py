from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer

class Main:
    def main(self):

        # Create instances for the classes that are used
        analyzer = GraphAnalyzer()
        utilityFunctions = UtilityFunctions()

        # Load dataset
        dataset = utilityFunctions.load_dataset(root="data/MUTAG", name="MUTAG", use_node_attr=True)   
        chain_distribution = {}

        for single_graph in dataset:
            graph, chain_starts, edges, neighbors, chain_lengths = analyzer.search_graph(single_graph)
            print(chain_lengths)

main = Main()
main.main()