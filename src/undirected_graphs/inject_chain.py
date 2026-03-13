from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer

def inject_chain(graph_id, chain_length):
    # Load the graph
    dataset = UtilityFunctions.load_dataset(root="data/REDDIT-BINARY", name="REDDIT-BINARY")
    graph = dataset[0]
    # Get tuple of (source, target) edges of graph

    graph_tuples = GraphAnalyzer().search_graph(graph)

    # Get list of dangling nodes¨

    print(graph_tuples)

inject_chain(graph_id=0, chain_length=3)

    